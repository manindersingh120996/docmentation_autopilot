"""
Neo4j implementation of the GraphDatabase ABC.

IMPLEMENTATION OVERVIEW:
------------------------
This class connects to a running Neo4j instance (local Docker in development,
Neo4j Aura in production) and implements every method defined in GraphDatabase.

The driver used is neo4j.GraphDatabase (the official synchronous Python driver).
We chose synchronous here to match the ABC's synchronous interface. The driver
internally manages a connection pool, so each method call acquires a session
from the pool, does its work, and returns it — we never manage raw connections.

WHY NEO4J FOR THIS PROBLEM:
    Our core question is "which documentation sections are affected when
    function X changes?" This requires traversing chains like:
        Function X  <--[CALLS]--  Function Y  <--[REFERENCES]--  DocSection Z
    
    In PostgreSQL this needs recursive CTEs that get slow and complex fast.
    Neo4j stores relationships as physical pointers on disk, so traversing
    one hop is a pointer dereference — O(1) per hop regardless of graph size.
    A 5-hop traversal across 1M nodes takes milliseconds.

SYNC VS ASYNC:
    The GraphDatabase ABC defines synchronous methods (no async/await).
    We use the sync neo4j.GraphDatabase driver accordingly.
    The driver manages a connection pool internally — each method borrows
    a session from the pool, runs its query, and returns the session.
    You never open/close raw connections yourself.

NODE ID STRATEGY:
    Neo4j assigns its own internal element IDs to every node. We ignore those.
    Instead, every node stores an "id" property that WE assign (UUID or a
    meaningful string like "func:auth:authenticate"). This ensures:
      - IDs survive database recreations (Neo4j internal IDs can reset)
      - IDs are portable if we swap to a different graph database
      - IDs are human-readable in logs and debug output
    All lookups use: MATCH (n {id: $id}) which hits our range index.

CYPHER INJECTION SAFETY:
    Most values are passed as $parameters (safe, parameterized).
    EXCEPTION: relationship types and node labels must be Cypher LITERALS —
    they cannot be parameterized. The language simply doesn't support it.
    We embed them via f-strings, which is safe ONLY because these values
    always come from our controlled internal constants ("REFERENCES", "CALLS"),
    never from raw user input. If that ever changes, add an allowlist validator.

    CYPHER QUICK REFERENCE (for reading this file):
-------------------------------------------------
  MATCH (n:Label {id: $id})        — find node with label and property filter
  CREATE (n:Label $props)          — create node with property map
  MERGE (n:Label {id: $id})        — create-or-match by identity key
  SET n += $props                  — merge-update properties (preserves unmentioned keys)
  DETACH DELETE n                  — delete node AND all its relationships
  -[r:TYPE]->                      — directed relationship of type TYPE
  -[r:TYPE|OTHER]->                — relationship of TYPE or OTHER
  *1..5                            — variable-length: 1 to 5 hops
  RETURN n, r, p                   — return nodes, relationships, paths
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase as Neo4jDriver
from neo4j.exceptions import (
    AuthError,
    Neo4jError,
    ServiceUnavailable,
)

from src.infrastructure.base.graph_db import (
    GraphDatabase,
    GraphDatabaseError,
    Node,
    NodeNotFoundError,
    Path,
    QueryError,
    Relationship
)

logger = logging.getLogger(__name__)

class Neo4jGraphDatabase(GraphDatabase):
    """
    Synchronous Neo4j implementation of GraphDatabase.

    Usage pattern:
        db = Neo4jGraphDatabase(uri, username, password)
        db.connect()        # call once at startup — verifies connectivity
        ...use the db...
        db.close()          # call once at shutdown — drains the pool
    """

    def __init__(self,
                 uri: str = "bolt://localhost:7687",
                 username: str = "neo4j",
                 password: str = "password",
                 database: str = "neo4j",
                 ):
        """
         Store configuration. Does NOT connect — call connect() explicitly.

        Separating __init__ from connect() is a deliberate pattern: it lets us
        construct the object during application startup (dependency injection)
        before the Docker container for Neo4j has finished initializing.
        The retry logic lives in the calling code, not the constructor.

        Args:
            uri:      bolt://localhost:7687 for local Docker.
                      neo4j+s://xxx.databases.neo4j.io for Neo4j Aura (TLS).
            database: "neo4j" is the only option on Community Edition.
                      Enterprise and Aura support named databases.

        """
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._driver = None # Set by connect()

    # ------------------------------------------------------------------
    # Lifecycle (not in ABC, but required for setup/teardown)
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Create the connection pool and verify the database is reachable.

        driver.verify_connectivity() runs a cheap internal health check.
        This gives us a fast, clear failure at startup rather than a confusing
        error on the first actual query.

        We then call _ensure_indexes() so that every node lookup by "id"
        uses an index scan (O log N) rather than a full node scan (O N).
        """
        try:
            self._driver = Neo4jDriver.driver(
                self._uri,
                auth = (self._username, self._password),
            )
            self._driver.verify_connectivity()
            self._ensure_indexes()
            logger.info(f"Connected to Neo4j at {self._uri}")
        
        except AuthError as e:
            raise GraphDatabaseError(
                f"Neo4j authentication failed for user '{self._username}'."
                f"Check NEO4J_PASSWORD in you .env.local"
            ) from e
        
        except ServiceUnavailable as e:
            raise GraphDatabaseError(
                f"Neo4j not reachable at {self._uri}. "
                f"Is Docker running? Try: docker-compose up -d neo4j"
            ) from e
        
    def close(self) -> None:
        """Release all pooled connections. Safe to call even if connect() was never called."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _session(self):
        """
        Return a driver session, raising clearly if connect() was skipped.

        Without this guard, skipping connect() would produce:
            AttributeError: 'NoneType' object has no attribute 'session'
        which is confusing. This guard produces:
            GraphDatabaseError: not connected. Call connect() first.
        which is actionable.
        
        """
        if self._driver is None:
            raise GraphDatabaseError(
                "Neo4jGraphDatabse not connected. Call connect() before using."
            )
        return self._driver.session(database = self._database)
    
    def _ensure_indexes(self) -> None:
        """
        Create property indexes so MATCH (n {id: $id}) is fast.

        Without an index, every lookup scans ALL nodes with a given label
        comparing the ID property one by one, O(N). with a range index,
        Neo4j uses a B-Tree lookup, O(log N). The difference is irrelevant
        at 100 nodes and catastrophic at 100,000.


        IF NOT EXISTS makes this idempotent — safe to run on every startup
        without erroring on an already-configured database.

        We create one index per label rather than a global property index
        for Neo4j 4.x compatibility (global indexes require Neo4j 5+).
        """
        index_statements = [
            "CREATE INDEX idx_function_id IF NOT EXISTS FOR (n:Function) ON (n.id)",
            "CREATE INDEX idx_docsection_id IF NOT EXISTS FOR (n:DocSection) ON (n.id)",
            "CREATE INDEX idx_module_id IF NOT EXISTS FOR (n:Module) ON (n.id)",
            "CREATE INDEX idx_file_id IF NOT EXISTS FOR (n:File) ON (n.id)"
        ]
        with self._session() as session:
            for stmt in index_statements:
                try:
                    session.run(stmt).consume()
                except Exception as e:
                    # Non-fatal: the index might already exist under a different
                    # name from a previous setup. Log and continue — the database
                    # still works, it's just potentially slower without the index.
                    logger.warning(f"Index creation skipped (non-fatal): {e}")

    def _node_from_neo4j(self,neo4j_node) -> Node:
        """
        Convert a raw neo4j driver Node object into our Node DTO

        The driver exposes:
            neo4j_node.labels       -frozenset of label strings, e.g. {"Function"}
            neo4j_node.items()   — iterator of (property_key, value) pairs
            neo4j_node.element_id — Neo4j's internal ID (we ignore this)

        We pop "id" out of the properties dict because it lives as a first-class
        field on Node, not inside the properties dictionary. Callers should access
        it as node.id, not node.properties["id"].
        """
        props = dict(neo4j_node.items())
        node_id = props.pop("id", None)
        return Node(
            id=node_id,
            labels=list(neo4j_node.labels),
            properties=props,
        )

    def _rel_from_neo4j(self, neo4j_rel, from_id: str, to_id: str) -> Relationship:
        """
        Convert a neo4j driver Relationship into our Relationship DTO.

        We need from_id and to_id passed in explicitly. The driver's
        neo4j_rel.start_node and neo4j_rel.end_node give us neo4j's internal
        element IDs, not our app-managed "id" properties. The caller extracts
        these from the surrounding MATCH pattern (e.g. RETURN a.id, b.id)
        which is more reliable.
        """
        props = dict(neo4j_rel.items())
        return Relationship(
            from_node_id=from_id,
            to_node_id=to_id,
            type=neo4j_rel.type,
            properties=props if props else None,
        )

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------


    def create_node(self, node: Node) -> str:
        """
        Create (or update) a node in the graph, returning its ID.

        WHY MERGE INSTEAD OF CREATE:
            Our pipeline uses at-least-once message delivery (RabbitMQ's guarantee).
            The same commit event might be processed twice. If we used CREATE,
            the second processing would either fail (if we had a uniqueness constraint)
            or silently create a duplicate node (if we didn't). Either outcome is bad.

            MERGE says: "find a node matching this pattern; if it exists update it,
            if it doesn't exist create it." This makes the operation idempotent —
            running it 10 times produces the same graph as running it once.

        WHY labels ARE IN THE f-string:
            Cypher does not allow label names as parameters. You cannot write:
                MERGE (n:$label {id: $id})   ← INVALID Cypher, will error
            Labels must be literals in the query. Since node.labels always comes
            from our controlled code (never raw user input), this f-string is safe.

        SET n += $props uses += (merge/overlay semantics):
            Existing properties NOT in the new dict are preserved.
            Properties IN the new dict are added or overwritten.
            This is correct for updates — we don't want to wipe properties
            we're not explicitly changing.
        """
        node_id = node.id or str(uuid.uuid4())
        # Multi-label syntax: (n:Label1:Label2) — joining with colon
        label_str = ":".join(node.labels) if node.labels else "Node"
        # Always store our app-managed id as a property on the node
        all_props = {**node.properties, "id": node_id}

        query = f"""
            MERGE (n:{label_str} {{id: $id}})
            SET n += $props
            RETURN n
        """
        try:
            with self._session() as session:
                result = session.run(query, id=node_id, props=all_props)
                result.single()  # consume the result
            logger.debug(f"Upserted node id={node_id} labels={node.labels}")
            return node_id
        except Neo4jError as e:
            raise GraphDatabaseError(f"Failed to create node: {e}") from e

    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Retrieve a node by its app-managed id property.

        Returns None (not an exception) when the node doesn't exist.
        This is the ABC's contract — get_node is a "maybe find" operation.
        Callers who need an error on missing nodes should check the return:
            node = db.get_node(id)
            if node is None:
                raise NodeNotFoundError(...)
        """
        query = "MATCH (n {id: $id}) RETURN n"
        try:
            with self._session() as session:
                result = session.run(query, id=node_id)
                record = result.single()
            if record is None:
                return None
            return self._node_from_neo4j(record["n"])
        except Neo4jError as e:
            raise GraphDatabaseError(f"Failed to get node {node_id}: {e}") from e

    def find_nodes(
        self,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Node]:
        """
        Find nodes by label and/or property filters, with AND semantics.

        Query construction works in three parts:
          1. MATCH clause — includes label filter if provided, bare (n) if not
          2. WHERE clause — one parameterized condition per property filter
          3. LIMIT clause — appended only if limit is provided

        We build WHERE conditions with indexed parameter names (val_0, val_1...)
        rather than the property keys themselves. This keeps the query shape
        stable across different calls, which lets Neo4j reuse its cached
        execution plan — a meaningful performance benefit for frequent lookups.
        """
        label_str = ":".join(labels) if labels else ""
        match_target = f"(n:{label_str})" if label_str else "(n)"

        where_parts = []
        params: Dict[str, Any] = {}
        if properties:
            for i, (key, val) in enumerate(properties.items()):
                param_name = f"val_{i}"
                # We use n.key syntax (property access) rather than putting
                # the filter in the MATCH pattern, to keep the query parameterized
                where_parts.append(f"n.{key} = ${param_name}")
                params[param_name] = val

        where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"MATCH {match_target} {where_clause} RETURN n {limit_clause}"
        try:
            with self._session() as session:
                result = session.run(query, **params)
                return [self._node_from_neo4j(r["n"]) for r in result]
        except Neo4jError as e:
            raise GraphDatabaseError(f"Failed to find nodes: {e}") from e

    def delete_node(self, node_id: str, cascade: bool = False) -> None:
        """
        Delete a node, optionally cascading to its relationships.

        We verify existence first to give a clear NodeNotFoundError.
        Without this check, Neo4j silently does nothing for a MATCH on
        a non-existent node — the caller would never know it failed.

        cascade=True → DETACH DELETE (removes node + all its relationships atomically)
        cascade=False → DELETE (fails if any relationships exist, matching ABC docs)
        """
        if self.get_node(node_id) is None:
            raise NodeNotFoundError(f"Node '{node_id}' not found")

        delete_clause = "DETACH DELETE n" if cascade else "DELETE n"
        query = f"MATCH (n {{id: $id}}) {delete_clause}"
        try:
            with self._session() as session:
                session.run(query, id=node_id).consume()
            logger.debug(f"Deleted node id={node_id}, cascade={cascade}")
        except Neo4jError as e:
            raise GraphDatabaseError(f"Failed to delete node {node_id}: {e}") from e

    # ------------------------------------------------------------------
    # Relationship operations
    # ------------------------------------------------------------------

    def create_relationship(self, relationship: Relationship) -> None:
        """
        Create a directed relationship between two existing nodes.

        Like create_node, we use MERGE for idempotency — calling this twice
        with the same (source, type, target) triple doesn't create duplicates.

        SET r += $props updates relationship properties with overlay semantics.
        If properties is None we pass an empty dict — SET r += {} is a no-op.

        We verify both nodes exist before the MATCH+MERGE query because
        Neo4j's MATCH clause on a non-existent node simply returns no results
        without raising an error — the relationship silently isn't created.
        Verifying first gives the caller a clear NodeNotFoundError.
        """
        if self.get_node(relationship.from_node_id) is None:
            raise NodeNotFoundError(
                f"Source node '{relationship.from_node_id}' not found"
            )
        if self.get_node(relationship.to_node_id) is None:
            raise NodeNotFoundError(
                f"Target node '{relationship.to_node_id}' not found"
            )

        props = relationship.properties or {}
        # relationship.type must be a Cypher literal — see module docstring
        query = f"""
            MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
            MERGE (a)-[r:{relationship.type}]->(b)
            SET r += $props
        """
        try:
            with self._session() as session:
                session.run(
                    query,
                    from_id=relationship.from_node_id,
                    to_id=relationship.to_node_id,
                    props=props,
                ).consume()
            logger.debug(
                f"Upserted relationship: {relationship.from_node_id}"
                f" -[{relationship.type}]-> {relationship.to_node_id}"
            )
        except Neo4jError as e:
            raise GraphDatabaseError(f"Failed to create relationship: {e}") from e

    def get_relationships(
        self,
        from_node_id: Optional[str] = None,
        to_node_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
    ) -> List[Relationship]:
        """
        Get relationships matching up to three optional filters.

        All three filters are optional and ANDed — any combination works:
            get_relationships(from_node_id="doc1")           # all outbound from doc1
            get_relationships(to_node_id="func1")            # all inbound to func1
            get_relationships(relationship_type="REFERENCES") # all REFERENCES anywhere
            get_relationships(from_node_id="doc1", to_node_id="func1") # specific edge

        The match pattern changes shape based on which filters are active.
        relationship_type becomes a literal in the pattern (cannot be parameterized).
        node IDs become $parameters (can and should be parameterized).
        """
        rel_type_str = f":{relationship_type}" if relationship_type else ""
        from_filter = "{id: $from_id}" if from_node_id else ""
        to_filter = "{id: $to_id}" if to_node_id else ""

        query = f"""
            MATCH (a {from_filter})-[r{rel_type_str}]->(b {to_filter})
            RETURN r, a.id AS from_id, b.id AS to_id
        """
        params: Dict[str, Any] = {}
        if from_node_id:
            params["from_id"] = from_node_id
        if to_node_id:
            params["to_id"] = to_node_id

        try:
            with self._session() as session:
                result = session.run(query, **params)
                return [
                    self._rel_from_neo4j(r["r"], r["from_id"], r["to_id"])
                    for r in result
                ]
        except Neo4jError as e:
            raise GraphDatabaseError(f"Failed to get relationships: {e}") from e

    def delete_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str,
    ) -> None:
        """
        Delete a specific directed relationship identified by (source, type, target).

        We identify by pattern rather than by a relationship ID because callers
        typically know "this doc no longer references this function" — they know
        the endpoints and type, not an internal relationship identifier.
        """
        query = f"""
            MATCH (a {{id: $from_id}})-[r:{relationship_type}]->(b {{id: $to_id}})
            DELETE r
        """
        try:
            with self._session() as session:
                session.run(query, from_id=from_node_id, to_id=to_node_id).consume()
            logger.debug(
                f"Deleted relationship: {from_node_id} -[{relationship_type}]-> {to_node_id}"
            )
        except Neo4jError as e:
            raise GraphDatabaseError(f"Failed to delete relationship: {e}") from e

    # ------------------------------------------------------------------
    # Graph traversal — the core reason we use Neo4j over PostgreSQL
    # ------------------------------------------------------------------

    def traverse(
        self,
        start_node_id: str,
        relationship_types: List[str],
        direction: str = "outgoing",
        max_depth: int = 10,
    ) -> List[Node]:
        """
        Traverse the graph from start_node_id, following relationship types.

        This is the heart of the documentation impact analysis. The typical call is:
            db.traverse(
                start_node_id="func_authenticate",
                relationship_types=["REFERENCES"],
                direction="incoming",
                max_depth=5
            )
        which answers: "find every node that REFERENCES authenticate, directly
        or through up to 5 hops of REFERENCES chains."

        HOW THE CYPHER PATTERN IS BUILT:
            relationship_types=["REFERENCES", "CALLS"] becomes :REFERENCES|CALLS
            The | in Cypher means OR for relationship types — traverse either type.

            max_depth=5 combined with *1.. gives the variable-length pattern:
            *1..5 means "one or more hops, up to five."
            We always use at least 1 (not 0) so the start node is excluded.

            Direction maps to Cypher arrow direction:
              "outgoing"  →  (start)-[r*1..N]->(other)   follow edges away
              "incoming"  →  (start)<-[r*1..N]-(other)   follow edges toward
              "both"      →  (start)-[r*1..N]-(other)    follow either way

        WHY DISTINCT:
            Multiple paths can lead to the same destination node. Without DISTINCT,
            Neo4j returns one result per path — the same node appears repeatedly.
            DISTINCT collapses this to one result per unique destination node.
        """
        if self.get_node(start_node_id) is None:
            raise NodeNotFoundError(f"Start node '{start_node_id}' not found")

        # Build the relationship type filter: "REFERENCES|CALLS"
        # The | operator in Cypher is OR for relationship types
        type_filter = "|".join(relationship_types) if relationship_types else ""
        rel_pattern = (
            f"[:{type_filter}*1..{max_depth}]" if type_filter
            else f"[*1..{max_depth}]"
        )

        if direction == "incoming":
            pattern = f"(start)<-{rel_pattern}-(other)"
        elif direction == "both":
            pattern = f"(start)-{rel_pattern}-(other)"
        else:  # "outgoing" is the default
            pattern = f"(start)-{rel_pattern}->(other)"

        query = f"""
            MATCH (start {{id: $start_id}})
            MATCH {pattern}
            WHERE other.id <> $start_id
            RETURN DISTINCT other
        """
        try:
            with self._session() as session:
                result = session.run(query, start_id=start_node_id)
                nodes = [self._node_from_neo4j(r["other"]) for r in result]
            logger.debug(
                f"Traversal from '{start_node_id}': {len(nodes)} nodes reachable "
                f"(types={relationship_types}, direction={direction}, max_depth={max_depth})"
            )
            return nodes
        except Neo4jError as e:
            raise GraphDatabaseError(f"Traversal failed from '{start_node_id}': {e}") from e

    def shortest_path(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 10,
    ) -> Optional[Path]:
        """
        Find the shortest path between two nodes.

        HOW Neo4j's shortestPath() WORKS:
            shortestPath() is a built-in Neo4j function that runs BFS internally.
            BFS guarantees the minimum-hop path. We don't implement BFS ourselves —
            we delegate to Neo4j's highly optimized native graph algorithm.

        PATH RECONSTRUCTION:
            The driver returns a neo4j Path object containing .nodes and
            .relationships sequences. The path structure is always:
                node[0], rel[0], node[1], rel[1], node[2], ...
            meaning nodes[i] and nodes[i+1] are connected by relationships[i].
            We use this invariant to pair consecutive nodes when reconstructing
            from_node_id and to_node_id for each Relationship DTO.

        RETURNS None (NOT raises) when no path exists:
            The ABC specifies Optional[Path] for this reason — disconnected nodes
            are a normal outcome, not an error condition. Callers decide whether
            disconnection is a problem for their use case.
        """
        if self.get_node(from_node_id) is None:
            raise NodeNotFoundError(f"Source node '{from_node_id}' not found")
        if self.get_node(to_node_id) is None:
            raise NodeNotFoundError(f"Target node '{to_node_id}' not found")

        type_filter = "|".join(relationship_types) if relationship_types else ""
        rel_pattern = (
            f"[:{type_filter}*..{max_depth}]" if type_filter
            else f"[*..{max_depth}]"
        )

        query = f"""
            MATCH (start {{id: $from_id}}), (end {{id: $to_id}})
            MATCH p = shortestPath((start)-{rel_pattern}-(end))
            RETURN p
        """
        try:
            with self._session() as session:
                result = session.run(query, from_id=from_node_id, to_id=to_node_id)
                record = result.single()

            if record is None:
                return None  # No path exists — normal outcome, not an error

            neo4j_path = record["p"]
            # Reconstruct our Node DTOs from the path's node sequence
            path_nodes = [self._node_from_neo4j(n) for n in neo4j_path.nodes]

            # Reconstruct Relationship DTOs using the alternating node-rel-node structure
            # path_nodes[i] and path_nodes[i+1] are the endpoints of relationships[i]
            path_rels = []
            for i, rel in enumerate(neo4j_path.relationships):
                path_rels.append(
                    self._rel_from_neo4j(rel, path_nodes[i].id, path_nodes[i + 1].id)
                )

            return Path(
                nodes=path_nodes,
                relationships=path_rels,
                length=len(path_rels),
            )
        except Neo4jError as e:
            raise GraphDatabaseError(
                f"Shortest path query failed ({from_node_id} → {to_node_id}): {e}"
            ) from e

    # ------------------------------------------------------------------
    # Raw query and maintenance
    # ------------------------------------------------------------------

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a raw Cypher query, returning results as plain Python dicts.

        This escape hatch exists for queries the convenience methods can't express —
        for example, a multi-hop transitive impact analysis with custom scoring
        logic, or an aggregation query for the analytics dashboard.

        result.data() serializes all records to plain Python dicts, so callers
        never need to import the neo4j library — the abstraction boundary is
        preserved even through this escape hatch. Neo4j objects don't leak out.
        """
        try:
            with self._session() as session:
                result = session.run(query, **(parameters or {}))
                return result.data()
        except Neo4jError as e:
            raise QueryError(
                f"Cypher query failed: {e}\nQuery preview: {query[:300]}"
            ) from e

    def clear(self) -> None:
        """
        Delete ALL nodes and relationships. Irreversible.

        MATCH (n) DETACH DELETE n is the canonical Neo4j wipe command.
        DETACH DELETE on every node cascades to every relationship,
        so one query handles the entire graph.

        Used by the test suite teardown fixture. Never called in production.
        """
        try:
            with self._session() as session:
                session.run("MATCH (n) DETACH DELETE n").consume()
            logger.warning("Graph database cleared — all data deleted")
        except Neo4jError as e:
            raise GraphDatabaseError(f"Failed to clear database: {e}") from e
