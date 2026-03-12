"""
Integration tests for Neo4jGraphDatabase.

These run against a REAL Neo4j instance — no mocking.
Neo4j's MERGE semantics, DETACH DELETE, and variable-length traversals
are complex enough that mocks give false confidence.

TWO WAYS TO RUN:
    1. Via pytest (full suite with coverage):
           pytest tests/integration/test_neo4j_graph_db.py -v

    2. As a script (quick manual check, like rabbitmq tests):
           python tests/integration/test_neo4j_graph_db.py

SETUP:
    docker-compose up -d neo4j
    # Wait ~15 seconds for Neo4j to initialize, then run.

ENVIRONMENT VARIABLES:
    NEO4J_URI       default: bolt://localhost:7687
    NEO4J_USERNAME  default: neo4j
    NEO4J_PASSWORD  default: devpassword  <-- match your docker-compose.yml

TEST ISOLATION:
    Every test prefixes node IDs with a unique RUN_ID for this session.
    Session teardown deletes all nodes with that prefix.
    Tests never interfere with each other or with pre-existing data.
"""

import os
import sys
import time
import uuid
from pathlib import Path

# Add project root to path — same pattern as rabbitmq test
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest

from src.infrastructure.local.neo4j_graph_db import Neo4jGraphDatabase
from src.infrastructure.base.graph_db import (
    GraphDatabaseError,
    Node,
    NodeNotFoundError,
    QueryError,
    Relationship,
)

# One unique short ID for this entire test session
RUN_ID = uuid.uuid4().hex[:6]


def nid(name: str) -> str:
    """Build an isolated node ID for this test session."""
    return f"t_{RUN_ID}_{name}"


def make_db() -> Neo4jGraphDatabase:
    """Create and connect a Neo4j instance using env vars or defaults."""
    db = Neo4jGraphDatabase(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "devpassword"),
    )
    db.connect()
    return db


# ---------------------------------------------------------------------------
# Session-scoped pytest fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def db():
    """
    One Neo4j connection for the whole pytest session.
    Session scope pays the ~1s connection cost once, not per test.
    """
    database = make_db()
    yield database

    # Teardown: remove all nodes created during this test session
    try:
        database.execute_query(
            "MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
            parameters={"prefix": f"t_{RUN_ID}_"},
        )
    except Exception:
        pass

    database.close()


# ---------------------------------------------------------------------------
# 1. Connectivity
# ---------------------------------------------------------------------------

class TestConnectivity:

    def test_basic_query_executes(self, db):
        """Confirm the connection is live with the simplest possible Cypher."""
        results = db.execute_query("RETURN 1 AS ping")
        assert results == [{"ping": 1}]

    def test_wrong_password_raises_graph_error(self):
        """
        AuthError from the driver must be re-raised as GraphDatabaseError.
        Application code should never see driver-specific exception types.
        """
        bad_db = Neo4jGraphDatabase(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="wrong_password_xyz_123",
        )
        with pytest.raises(GraphDatabaseError, match="authentication failed"):
            bad_db.connect()


# ---------------------------------------------------------------------------
# 2. Node CRUD
# ---------------------------------------------------------------------------

class TestNodeCrud:

    def test_create_and_retrieve_node(self, db):
        """Full round-trip: create a node, retrieve it, confirm data integrity."""
        node_id = nid("create_get")
        returned_id = db.create_node(Node(
            id=node_id,
            labels=["Function"],
            properties={"name": "authenticate", "file": "src/auth.py"},
        ))

        assert returned_id == node_id

        retrieved = db.get_node(node_id)
        assert retrieved is not None
        assert retrieved.id == node_id
        assert "Function" in retrieved.labels
        assert retrieved.properties["name"] == "authenticate"

    def test_create_without_id_generates_one(self, db):
        """When node.id is None, create_node() must generate and return a non-empty ID."""
        generated_id = db.create_node(Node(
            labels=["DocSection"],
            properties={"title": "Overview"},
        ))
        assert generated_id and len(generated_id) > 0

        node = db.get_node(generated_id)
        assert node is not None
        assert node.id == generated_id

    def test_create_node_with_multiple_labels(self, db):
        """A node can simultaneously belong to multiple label categories."""
        node_id = nid("multi_label")
        db.create_node(Node(id=node_id, labels=["Function", "PublicAPI"], properties={}))

        node = db.get_node(node_id)
        assert "Function" in node.labels
        assert "PublicAPI" in node.labels

    def test_get_node_returns_none_for_missing(self, db):
        """ABC contract: get_node returns None (not raises) when not found."""
        assert db.get_node("id_that_cannot_possibly_exist_zzzz") is None

    def test_create_node_is_idempotent(self, db):
        """
        Same id processed twice must not create duplicates — MERGE semantics.
        Second call updates properties (version becomes 2).
        """
        node_id = nid("idempotent")
        db.create_node(Node(id=node_id, labels=["Module"], properties={"version": 1}))
        db.create_node(Node(id=node_id, labels=["Module"], properties={"version": 2}))

        nodes = db.find_nodes(labels=["Module"], properties={"version": 2})
        matching = [n for n in nodes if n.id == node_id]
        assert len(matching) == 1

    def test_find_nodes_by_label_and_property(self, db):
        """find_nodes filters correctly — label AND property, Functions only."""
        db.create_node(Node(id=nid("fn_a"), labels=["Function"], properties={"run": RUN_ID}))
        db.create_node(Node(id=nid("fn_b"), labels=["Function"], properties={"run": RUN_ID}))
        db.create_node(Node(id=nid("doc_a"), labels=["DocSection"], properties={"run": RUN_ID}))

        functions = db.find_nodes(labels=["Function"], properties={"run": RUN_ID})
        ids = {n.id for n in functions}

        assert nid("fn_a") in ids
        assert nid("fn_b") in ids
        assert nid("doc_a") not in ids

    def test_find_nodes_returns_empty_list_when_none_match(self, db):
        """No match → empty list, never None, never an exception."""
        result = db.find_nodes(properties={"name": "function_does_not_exist_zzz"})
        assert result == []

    def test_delete_node_with_cascade(self, db):
        """cascade=True removes node AND all attached relationships atomically."""
        node_id = nid("del_cascade")
        other_id = nid("del_cascade_other")
        db.create_node(Node(id=node_id, labels=["Function"], properties={}))
        db.create_node(Node(id=other_id, labels=["DocSection"], properties={}))
        db.create_relationship(Relationship(
            from_node_id=other_id, to_node_id=node_id, type="REFERENCES"
        ))

        db.delete_node(node_id, cascade=True)

        assert db.get_node(node_id) is None
        assert db.get_relationships(to_node_id=node_id) == []

    def test_delete_nonexistent_node_raises(self, db):
        """Attempting to delete a missing node must raise NodeNotFoundError."""
        with pytest.raises(NodeNotFoundError):
            db.delete_node("definitely_does_not_exist_abc_999")


# ---------------------------------------------------------------------------
# 3. Relationship CRUD
# ---------------------------------------------------------------------------

class TestRelationshipCrud:

    @pytest.fixture(autouse=True)
    def setup(self, db):
        p = f"t_{RUN_ID}_rel_{uuid.uuid4().hex[:4]}"
        self.doc_id = f"{p}_doc"
        self.func_a  = f"{p}_func_a"
        self.func_b  = f"{p}_func_b"
        self.db = db

        db.create_node(Node(id=self.doc_id, labels=["DocSection"], properties={}))
        db.create_node(Node(id=self.func_a,  labels=["Function"],   properties={}))
        db.create_node(Node(id=self.func_b,  labels=["Function"],   properties={}))

    def test_create_and_retrieve_relationship(self):
        self.db.create_relationship(Relationship(
            from_node_id=self.doc_id,
            to_node_id=self.func_a,
            type="REFERENCES",
            properties={"line_number": 42},
        ))

        rels = self.db.get_relationships(
            from_node_id=self.doc_id, relationship_type="REFERENCES"
        )
        matching = [r for r in rels if r.to_node_id == self.func_a]
        assert len(matching) == 1
        assert matching[0].properties["line_number"] == 42

    def test_create_relationship_missing_source_raises(self):
        with pytest.raises(NodeNotFoundError, match="Source"):
            self.db.create_relationship(Relationship(
                from_node_id="ghost_xyz", to_node_id=self.func_a, type="REFERENCES"
            ))

    def test_create_relationship_missing_target_raises(self):
        with pytest.raises(NodeNotFoundError, match="Target"):
            self.db.create_relationship(Relationship(
                from_node_id=self.doc_id, to_node_id="ghost_xyz", type="REFERENCES"
            ))

    def test_create_relationship_is_idempotent(self):
        """Creating (source, type, target) three times must produce exactly one relationship."""
        for _ in range(3):
            self.db.create_relationship(Relationship(
                from_node_id=self.func_a, to_node_id=self.func_b, type="CALLS"
            ))

        rels = self.db.get_relationships(
            from_node_id=self.func_a, to_node_id=self.func_b, relationship_type="CALLS"
        )
        assert len(rels) == 1

    def test_get_relationships_inbound(self):
        self.db.create_relationship(Relationship(
            from_node_id=self.doc_id, to_node_id=self.func_a, type="REFERENCES"
        ))
        inbound = self.db.get_relationships(to_node_id=self.func_a)
        assert any(r.from_node_id == self.doc_id for r in inbound)

    def test_delete_relationship(self):
        self.db.create_relationship(Relationship(
            from_node_id=self.doc_id, to_node_id=self.func_a, type="REFERENCES"
        ))
        self.db.delete_relationship(self.doc_id, self.func_a, "REFERENCES")

        rels = self.db.get_relationships(
            from_node_id=self.doc_id, to_node_id=self.func_a, relationship_type="REFERENCES"
        )
        assert rels == []


# ---------------------------------------------------------------------------
# 4. Traversal
# ---------------------------------------------------------------------------

class TestTraversal:
    """
    Graph built for these tests:

        checkout_guide (DocSection)
               |
               REFERENCES
               v
        calculate_price (Function)
               |
               CALLS
               v
        apply_discount (Function)
               |
               CALLS
               v
        get_user_tier (Function)   <-- the changed function

    Changing get_user_tier must bubble up and flag checkout_guide as stale,
    even though the guide never mentions get_user_tier directly.
    """

    @pytest.fixture(autouse=True)
    def build_graph(self, db):
        p = f"t_{RUN_ID}_trav_{uuid.uuid4().hex[:4]}"
        self.db = db

        self.tier_id  = f"{p}_get_user_tier"
        self.disc_id  = f"{p}_apply_discount"
        self.price_id = f"{p}_calculate_price"
        self.checkout = f"{p}_checkout_guide"
        self.disc_doc = f"{p}_discount_policy"

        for node_id, label, name in [
            (self.tier_id,  "Function",   "get_user_tier"),
            (self.disc_id,  "Function",   "apply_discount"),
            (self.price_id, "Function",   "calculate_price"),
            (self.checkout, "DocSection", "Checkout Guide"),
            (self.disc_doc, "DocSection", "Discount Policy"),
        ]:
            db.create_node(Node(id=node_id, labels=[label], properties={"name": name}))

        db.create_relationship(Relationship(self.price_id, self.disc_id,  "CALLS"))
        db.create_relationship(Relationship(self.disc_id,  self.tier_id,  "CALLS"))
        db.create_relationship(Relationship(self.checkout, self.price_id, "REFERENCES"))
        db.create_relationship(Relationship(self.disc_doc, self.disc_id,  "REFERENCES"))

    def test_direct_reference_found(self):
        nodes = self.db.traverse(
            start_node_id=self.disc_id,
            relationship_types=["REFERENCES"],
            direction="incoming",
            max_depth=1,
        )
        assert self.disc_doc in {n.id for n in nodes}

    def test_transitive_impact_detected(self):
        """
        THE key test — validates the core reason for using Neo4j.
        checkout_guide must be found even though it is 3 hops from get_user_tier.
        """
        nodes = self.db.traverse(
            start_node_id=self.tier_id,
            relationship_types=["CALLS", "REFERENCES"],
            direction="incoming",
            max_depth=5,
        )
        ids = {n.id for n in nodes}

        assert self.disc_id   in ids, "apply_discount (1 hop CALLS)"
        assert self.price_id  in ids, "calculate_price (2 hops CALLS)"
        assert self.disc_doc  in ids, "discount_policy (refs apply_discount)"
        assert self.checkout  in ids, "checkout_guide (3 hops, must be found transitively)"

    def test_max_depth_limits_traversal(self):
        nodes = self.db.traverse(
            start_node_id=self.tier_id,
            relationship_types=["CALLS"],
            direction="incoming",
            max_depth=1,
        )
        ids = {n.id for n in nodes}
        assert self.disc_id  in ids
        assert self.price_id not in ids

    def test_nonexistent_start_raises(self):
        with pytest.raises(NodeNotFoundError):
            self.db.traverse("does_not_exist_xyz_999", ["REFERENCES"])

    def test_isolated_node_returns_empty_list(self):
        isolated = f"t_{RUN_ID}_isolated_{uuid.uuid4().hex[:4]}"
        self.db.create_node(Node(id=isolated, labels=["Function"], properties={}))
        result = self.db.traverse(isolated, ["REFERENCES"], direction="incoming")
        assert result == []


# ---------------------------------------------------------------------------
# 5. Shortest Path
# ---------------------------------------------------------------------------

class TestShortestPath:

    @pytest.fixture(autouse=True)
    def build_chain(self, db):
        """Linear chain: A -[CALLS]-> B -[CALLS]-> C, plus an isolated node."""
        p = f"t_{RUN_ID}_path_{uuid.uuid4().hex[:4]}"
        self.a        = f"{p}_a"
        self.b        = f"{p}_b"
        self.c        = f"{p}_c"
        self.isolated = f"{p}_isolated"
        self.db = db

        for node_id in [self.a, self.b, self.c, self.isolated]:
            db.create_node(Node(id=node_id, labels=["Function"], properties={}))
        db.create_relationship(Relationship(self.a, self.b, "CALLS"))
        db.create_relationship(Relationship(self.b, self.c, "CALLS"))

    def test_direct_path_length_one(self):
        path = self.db.shortest_path(self.a, self.b)
        assert path is not None
        assert path.length == 1

    def test_transitive_path_length_two(self):
        path = self.db.shortest_path(self.a, self.c)
        assert path is not None
        assert path.length == 2

    def test_disconnected_returns_none(self):
        assert self.db.shortest_path(self.a, self.isolated) is None

    def test_path_structure_invariant(self):
        """len(nodes) == length + 1, len(relationships) == length. Always."""
        path = self.db.shortest_path(self.a, self.c)
        assert len(path.nodes)         == path.length + 1
        assert len(path.relationships) == path.length

    def test_missing_node_raises(self):
        with pytest.raises(NodeNotFoundError):
            self.db.shortest_path(self.a, "ghost_node_xyz_999")


# ---------------------------------------------------------------------------
# 6. Raw Cypher
# ---------------------------------------------------------------------------

class TestExecuteQuery:

    def test_simple_return(self, db):
        assert db.execute_query("RETURN 42 AS answer")[0]["answer"] == 42

    def test_parameterized_query(self, db):
        node_id = nid("raw_query")
        db.create_node(Node(id=node_id, labels=["Module"], properties={"version": "2.0"}))
        results = db.execute_query(
            "MATCH (n {id: $id}) RETURN n.version AS v",
            parameters={"id": node_id},
        )
        assert results[0]["v"] == "2.0"

    def test_invalid_cypher_raises_query_error(self, db):
        with pytest.raises(QueryError):
            db.execute_query("THIS IS NOT VALID CYPHER !!!")


# ---------------------------------------------------------------------------
# 7. clear()
# ---------------------------------------------------------------------------

class TestClear:

    def test_clear_removes_all_nodes(self):
        """Uses a separate connection so clear() doesn't wipe the shared session db."""
        isolated_db = make_db()
        try:
            sentinel_id = f"clear_sentinel_{uuid.uuid4().hex[:6]}"
            isolated_db.create_node(Node(id=sentinel_id, labels=["Test"], properties={}))
            isolated_db.clear()
            assert isolated_db.get_node(sentinel_id) is None
            print("✓ clear() removed all nodes")
        finally:
            isolated_db.close()


# ---------------------------------------------------------------------------
# Standalone functions for __main__ block (mirrors rabbitmq test pattern)
# ---------------------------------------------------------------------------

def test_connection_manual(db: Neo4jGraphDatabase):
    result = db.execute_query("RETURN 1 AS ping")
    assert result == [{"ping": 1}]
    print("✓ Connection verified")


def test_node_roundtrip_manual(db: Neo4jGraphDatabase):
    node_id = f"manual_{uuid.uuid4().hex[:8]}"
    db.create_node(Node(id=node_id, labels=["Function"], properties={"name": "test_func"}))
    node = db.get_node(node_id)
    assert node is not None
    assert node.properties["name"] == "test_func"
    db.delete_node(node_id, cascade=True)
    assert db.get_node(node_id) is None
    print(f"✓ Node round-trip: create → get → delete (id={node_id})")


def test_relationship_manual(db: Neo4jGraphDatabase):
    p = f"manual_{uuid.uuid4().hex[:6]}"
    doc_id  = f"{p}_doc"
    func_id = f"{p}_func"

    db.create_node(Node(id=doc_id,  labels=["DocSection"], properties={"title": "Guide"}))
    db.create_node(Node(id=func_id, labels=["Function"],   properties={"name": "do_thing"}))
    db.create_relationship(Relationship(
        from_node_id=doc_id, to_node_id=func_id,
        type="REFERENCES", properties={"line_number": 10}
    ))

    rels = db.get_relationships(from_node_id=doc_id, relationship_type="REFERENCES")
    assert len(rels) == 1
    assert rels[0].properties["line_number"] == 10

    # Cleanup
    db.delete_node(doc_id,  cascade=True)
    db.delete_node(func_id, cascade=True)
    print(f"✓ Relationship round-trip: create → get → delete")


def test_traversal_manual(db: Neo4jGraphDatabase):
    """
    Build the full chain and confirm transitive impact detection.

        checkout_guide -[REFERENCES]-> calculate_price
                                            -[CALLS]-> apply_discount
                                                           -[CALLS]-> get_user_tier

    traverse("get_user_tier", incoming) must return checkout_guide.
    """
    p = f"manual_{uuid.uuid4().hex[:6]}"

    tier_id   = f"{p}_tier"
    disc_id   = f"{p}_disc"
    price_id  = f"{p}_price"
    checkout  = f"{p}_checkout"

    for node_id, label in [
        (tier_id,  "Function"),
        (disc_id,  "Function"),
        (price_id, "Function"),
        (checkout, "DocSection"),
    ]:
        db.create_node(Node(id=node_id, labels=[label], properties={}))

    db.create_relationship(Relationship(price_id, disc_id,  "CALLS"))
    db.create_relationship(Relationship(disc_id,  tier_id,  "CALLS"))
    db.create_relationship(Relationship(checkout, price_id, "REFERENCES"))

    nodes = db.traverse(
        start_node_id=tier_id,
        relationship_types=["CALLS", "REFERENCES"],
        direction="incoming",
        max_depth=5,
    )
    found_ids = {n.id for n in nodes}
    assert checkout in found_ids, "checkout_guide must be found transitively"
    print(f"✓ Transitive traversal: found {len(nodes)} affected nodes including checkout_guide")

    # Cleanup
    for node_id in [tier_id, disc_id, price_id, checkout]:
        db.delete_node(node_id, cascade=True)


def test_invalid_json_manual(db: Neo4jGraphDatabase):
    """Confirm QueryError on bad Cypher — mirrors rabbitmq's invalid json test."""
    try:
        db.execute_query("THIS IS NOT VALID CYPHER")
        assert False, "Should have raised QueryError"
    except QueryError:
        print("✓ Invalid Cypher raises QueryError correctly")


# ---------------------------------------------------------------------------
# __main__ block — run as script for quick manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Neo4jGraphDatabase Integration Tests")
    print("=" * 70)
    print()
    print("Ensure Neo4j is running: docker-compose up -d neo4j")
    print()

    db_instance = make_db()

    try:
        test_connection_manual(db_instance)
        print()

        test_node_roundtrip_manual(db_instance)
        print()

        test_relationship_manual(db_instance)
        print()

        test_traversal_manual(db_instance)
        print()

        test_invalid_json_manual(db_instance)
        print()

        print("=" * 70)
        print("✓ All manual tests passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise

    finally:
        db_instance.close()
