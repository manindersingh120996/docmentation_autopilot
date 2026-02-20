"""
Docstring for doc-auto-pilot.src.infrastructure.base.graph_db
--------------------------------------------------------------------------------

Abstract base class for graph database implementations.

A GraphDatabase stores nodes (entities) and relationships (edges) between
them, optimized for traversal queries , follwoing connections across many 
hops efficiently.

WHY THIS EXISTS:
    Documentation has complex dependencies. A function may be documented in
    three different places. A module's README references ten functions.
    A tutorial depends on five API docs. When 'authenticate()' changes,
    we need to find ALL  documentation that (directly or transitively)
    references it.

    This is a graph traversal problem, Relational databases are poor
    at it (recursive joins are slow and complex). Graph Datbases handle it
    natively in milli-seconds.

WHAT THIS ABSTRACTS:
    Local development: Neo4j (running in Docker)
    Production:        Neo4j Aura (managed cloud) or another graph DB

EXAMPLE GRAPH IN OUR SYSTEM:
    (authenticate: Function) --[DEFINED_IN]--> (auth.py: File)
    (UserGuide: DocSection) --[REFERENCES]--> (authenticate: Function)
    (APIReference: DocSection) --[REFERENCES]--> (authenticate: Function)
    (QuickStart: DocSection) --[REFERENCES]--> (UserGuide: DocSection)

    Query: "authenticate changed — which docs are affected?"
    Traversal: follow REFERENCES edges backwards from authenticate,
               then follow REFERENCES edges from found docs, up to depth 5.
    Result: [UserGuide, APIReference, QuickStart] — all need updating.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """
    A node in the graph, represents any entitiy in our system

    Labels are like types/categories. A node can have multiple labels.
    Properties are the data stored on the node.

    Examples of nodes in our documentation dependency graph:
        Node(labels=["Function"], properties={"name": "authenticate", "file": "auth.py"})
        Node(labels=["DocSection"], properties={"title": "User Guide", "path": "docs/guide.md"})
        Node(labels=["Module"], properties={"name": "torch.nn.functional"})

    Attributes:
        id:         Unique identifier. If None at creation, auto-generated.
                    After creation, always populated.
        labels:     List of type labels (e.g., ["Function", "CodeElement"]).
                    Allows nodes to belong to multiple categories.
        properties: Key-value data stored on this node.
    """
    id: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Relationship:
    """
    Docstring for Relationship
    ------------------------------------------------

    A directed edge between two nodes , represents a relationship

    Relationship are alwyas DIRECTIONAL:  from node -> to_node
    The direction encodes meaning:
        (DocSection) --[REFERENCES]--> (Function)   means: doc references code
        (Function) --[REFERENCES]--> (DocSection)   means: code references doc (different!)

    Examples in our system:
        Relationship(from_node_id="doc1", to_node_id="func1", type="REFERENCES")
        Relationship(from_node_id="func1", to_node_id="func2", type="CALLS")
        Relationship(from_node_id="module1", to_node_id="func1", type="CONTAINS")

    Attributes:
        from_node_id:  ID of the source node (relationship starts here)
        to_node_id:    ID of the target node (relationship points here)
        type:          Relationship type in UPPER_SNAKE_CASE by convention.
                       Examples: "REFERENCES", "DEPENDS_ON", "CALLS", "EXPLAINS"
        properties:    Optional data on the relationship itself.
                       Example: {"line_number": 42, "confidence": 0.95}


    """
    from_node_id: str
    to_node_id: str
    type: str
    properties: Optional[Dict[str,Any]] = None


@dataclass
class Path:
    """
    Docstring for Path

    A traversal path through the graph, a sequence of nodes and relationships.

    Represents the route found by shortest_path() or complex traversal.
    Useful for understanding HOW two nodes are connected, not just ID they are.

    Example path: DocSection → (REFERENCES) → Function → (CALLS) → OtherFunction
        nodes:         [DocSection, Function, OtherFunction]
        relationships: [(REFERENCES), (CALLS)]
        length:        2 (number of hops)

    Attributes:
        nodes:         Ordered list of nodes visited (length = path.length + 1)
        relationships: Ordered list of relationships traversed (length = path.length)
        length:        Number of relationship hops in this path
    """
    nodes: List[Node]
    relationships: List[Relationship]
    length: int

# ---------------------------------------------------------------------------
# Exception Hierarchy
# ---------------------------------------------------------------------------

class GraphDatabaseError(Exception):
    """Base exception for all graph databse erors."""
    pass

class NodeNotFoundError(GraphDatabaseError):
    """
    Docstring for NodeNotFoundError

    Raised when an operation refercens a node ID that doesn't exists.

    This is distinct from "no results found" — it means a specific ID
    was requested and it genuinely doesn't exist in the graph.
    """
    pass

class QueryError(GraphDatabaseError):
    """
    Docstring for QueryError

    Common causes:
        - Syntax error in the query string
        - Referencing non-existent properties and labels
        - Type mismatch in query paramters
    """
    pass

# ---------------------------------------------------------------------------
# The Abstract Base Class
# ---------------------------------------------------------------------------

class GraphDatabse(ABC):
    """
    Abstract interfaces for graph database implementations.

    CORE CONCEPT:
    Graphs have NODES (things) and RELATIONSHIPS (connections between things).
    Graph databases excel at traversal: given a node A, find all the nodes
    recahble by following specific relationship types upto N hops.

    This is fundamentally different from relational databases. SQL can do graph
    traversals but needs recursive CTEs that become slow and complex.
    Graph databases handle traversals natively and efficiently.

    TWO LEVELS OF API:
        1. Convenience methods: create_node(), find_nodes(), traverse() etc.
            Use these for common operations, they're prtable accross implementations.

        2. Raw query: execute_query() accepts native query language (Cypher for Neo4j).
            Use this for complex patterns the convenience methods can't express.
            Trade-off: less portable ( Cypher is Neo4j specific)

    CYPHER QUERY LANGUAGE (used in execuite_query for Neo4j):
        Cypher uses ASCII-art patterns to describe graph structure:
            MATCH (n:Function)-[:CALLS] ->(m:Function)
            meaning: "Find function nodes n that CALL other Function nodes m"

        Parameters prevent injection attacks (like in SQL parameters):
            query = "MATCH (n:Function) WHERE n.name = $name RETURN n"
            graph_db.execute_query(query, {"name": "authenticate"})

    """

    @abstractmethod
    def create_node(self,
                    node: Node
                    ) -> str:
        """
        Docstring for create_node
        -----------------------------
        Create a new node in the graphs.

        If node.id is None, the database generates a Unique ID.
        If node.id is provided, it's used as the ID (or merged if ti exists,
        depending on implementation)

        Args:
            node: Node with labels and properties. id may be None.

        Returns:
            The ID of the created node (generated or user-provided).

        Raises:
            GraphDatabaseError: If node creation fails.
        
            
        Example:
            node_id = graph_db.create_node(Node(
            labels = ["Function","CodeElement"],
            properties={
                    "name": "authenticate",
                    "file":"sc/auth/service.py",
                    "line":42,
                    "signature": "def authenticate(user_id: str) -> bool"
                    }
                    )
                    )
        """
        pass

    @abstractmethod
    def create_relationship(self, relationship: Relationship)->None:
        """
        Create a directed relationship between two existing nodes.

        Both nodes must exist before  calling this. The 
        relationship is directed: from_node_id -> to_node_id.

        Args:
            relationship: Relationship specifying source, target,type, and
            optional properties.

        Raises:
            NodeNotFoundError:  If either node ID doesn't exist.
            GraphDatabseError: If relationship creation fails.

        Example: 
            graph_dp.craete_relationship(Relationship(
                from_node_id = "doc_section_456",
                to_node_id="func_authenticate_123",
                type = "REFERENCES",
                properties = {"line_in_doc":17,
                            "confidence":0.98}
                            ))
        
        """
        pass

    @abstractmethod
    def find_nodes(
        self,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str,Any]] = None,
        limit: Optional[int] = None
    ) -> List[Node]:
        """
        Find nodes matching label and property ciriteria.

        This is a convenience method for simple lookups, For
        complex pattern matching, use execute_query() with Cypher.

        ALL condituions are ANDed together:
            - All labels must be present on the node
            - All properties mush exactly match

        Args:
            labels:     Filter by labels. All listed labels must be present.
                        None = match any labels.
            properties: Filter by exact property values.
                        None = match any properties.
            limit:      Maximum results to return. None = no limit.

        Returns:
            List of matching nodes. Empty list if none found.

        Example: 
            # find all functions nodes in auth.py
            funcs = graph_db.find_nodes(
                        labels = ["Function"],
                        properties = {"file": "src/auth/service.py"}
                        )
        """
        pass

    @abstractmethod
    def get_node( self, node_id: str) -> Optional[Node]:
        """
        Retrieve a single node by its ID.

        Args:
            node_id: ID of the node to retrieve.

        RETURNS:
            Node object if found. None if not found.

        Raises:
            GraphDatabaseError: If retrieval fails (NOT for "not found")
        """
        pass

    @abstractmethod
    def get_relationships(
        self,
        from_node_id: Optional[str] = None,
        to_node_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
    ) -> List[Relationship]:
        """
        Get relationships matching filter criteria.

        All specified filters are ANDed. Unspecified parameters match anything.

        Args:
            from_node_id:      Filter by source node. None = any source.
            to_node_id:        Filter by target node. None = any target.
            relationship_type: Filter by type. None = any type.

        Returns:
            List of matching relationships. Empty list if none found.

        Examples:
            # All relationships FROM a node (what does this node reference?)
            graph_db.get_relationships(from_node_id="doc123")

            # All relationships TO a node (what references this function?)
            graph_db.get_relationships(to_node_id="func_authenticate")

            # All REFERENCES relationships (regardless of nodes)
            graph_db.get_relationships(relationship_type="REFERENCES")

            # Specific relationship between specific nodes
            graph_db.get_relationships(
                from_node_id="doc123",
                to_node_id="func456",
                relationship_type="REFERENCES"
            )
        """
        pass

    @abstractmethod
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str,Any]] = None,

    ) -> List[Dict[str,Any]]:
        """
        Execute a native graph query (Cypher for Neo4j).

        This is the most powerful and flexible method , use it for
        complex patterns that convenience methods can't express. The
        tradeoff is that the query language is  implementation-specific (Cypher != Gremlin)

        Always use parameters for variable values. Never format variables
        directly into the query string. (SQL/Cypher injection risk).

        Args:
            query:      Native query string (Cypher for Neo4j).
            parameters: Values to substitute into the query safely.
                        Access in Cypher with $parameter_name syntax.

        Returns:
            List of result records as dictionaries.
            Keys are the aliases used in the RETURN clause.
            Values can be primitive types, Node objects, Relationship objects, etc.

        Raises:
            QueryError:         If query syntax is invalid or execution fails.
            GraphDatabaseError: If connection fails.

        Example: find all the docs affected by a file change
            results = graph_db.execute_query(
                    query = '''
                        MATCH (d:DocSection)-[:REFERENCES]->(f:Function)
                        WHERE f.file = $file_path
                        RETURN d.id AS doc_id, d.title AS title, f.name AS function_name
                        ORDER BY d.title
                    ''',
                    paramters={"file_path":"src/auth/service.py"}
                    )
            for r in results:
            print(f"Doc '{r['title']}' references {r['function_name']}")

        """
        pass

    @abstractmethod
    def traverse(
        self,
        start_node_id: str,
        relationship_types: List[str],
        direction: str = "outgoing",
        max_depth: int = 10
    ) -> List[Node]:
        """
        Traverse the graph from a starting node, following relationship types.

        This is the core operation for our documentation impact analysis:
        "Given that this function changed, find all documentation that
        references it (directly or transitively)."

        Args:
            start_node_id:      ID of the node to start from.
            relationship_types: Types of relationships to follow.
                                Only these types are traversed; others are ignored.
                                Example: ["REFERENCES", "DEPENDS_ON"]
            direction:          "outgoing" — follow edges AWAY from start node
                                "incoming" — follow edges TOWARD start node
                                "both"     — follow edges in either direction
                                For impact analysis, use "incoming" to find
                                what references the changed code.
            max_depth:          Maximum hops to traverse. Prevents infinite loops
                                in cyclic graphs and controls performance.
                                Our dependency graph is unlikely to need > 10 hops.

        Returns:
            All nodes reachable from start_node_id following the given constraints.
            May include the start node itself (implementation-specific).
            Order is not guaranteed.

        Raises:
            NodeNotFoundError:  If start_node_id doesn't exist.
            GraphDatabaseError: If traversal fails.

        Example: Find all docs that refernce authenticate() (directly or via other docs) :

            affected_docs = graph_db.traverse(
                        start_node_id = "func_authenticate_1234",
                        relationship_types = ["REFERENCES"],
                        direction = "incoming",
                        max_depth = 5
                        )
        """
        pass

    @abstractmethod
    def shortest_path(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 10
    ) -> Optional[Path]:
        """
        Find the shortest path between two nodes.

        Useful for understanding HOW two nodes are related, not just whether
        they are. For example: how is a high-lever tutorial connected to a 
        low-lever utility function ?

        Args:
            from_node_id:      Starting node ID.
            to_node_id:        Ending node ID.
            relationship_types: Types of relationships to traverse.
                               None = traverse any relationship type.
            max_depth:          Maximum path length to consider.
                               Paths longer than this are not returned.

        Returns:
            Path object if a path exists within max_depth.
            None if no path exists (nodes are not connected within depth).

        Raises:
            NodeNotFoundError:  If either node doesn't exist.
            GraphDatabaseError: If path finding fails.


        Example :
            path = graph_db.shortest_path("tutorial_doc_123", "func_low_level_345")
            if path:
                print(f"Connected in {path.length} hops:")
                for rel in path.relationships:
                    print(f" via {rel.types}")
            else:
                print("Not connected within max depth")

        """
        pass

    @abstractmethod
    def delete_node(
        self,
        node_id: str,
        cascade: bool = False,
    ) -> None:
        """
        Delete a node from the graph

        Args:
            node_id: ID of the node to delete.
            cascade: False (default): raise error if node has relationships.
                     True: delete the node AND alll its relationships.
                     Always consider cascade carefully — deleting relationships
                     may leave the graph in an inconsistent state.

        Raises:
            NodeNotFoundError:  If node doesn't exist.
            GraphDatabaseError: If node has relationships and cascade=False,
                               or if deletion fails for other reasons.

        """
        pass

    @abstractmethod
    def delete_relationship(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str
    ) -> None:
        """
        Delete a specific relationship between two nodes.

        Args:
            from_node_id:      Source node ID.
            to_node_id:        Target node ID.
            relationship_type: Type of relationship to delete.

        Raises:
            GraphDatabaseError: If relationship doesn't exist or deletion fails.

        """
        pass
    
    @abstractmethod
    def clear(self,) -> None:
        """
        Delete ALL nodes and ALL relationships in the databses.
        
        WARNING: Completely wipes the graph. Irreversible.
        Use ONLY in tests or when intentionally rebuilding from scratch.

        Raises:
            GraphDatabaseError: If clearing fails.
        """
        pass