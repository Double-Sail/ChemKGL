# src/neo4j_utils.py
from typing import List, Dict, Any
try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None
import re

class Neo4jUtils:
    def __init__(self, uri: str, user: str, password: str):
        if GraphDatabase is None:
            raise RuntimeError("neo4j-driver not installed or failed to import.")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        try:
            self.driver.close()
        except Exception:
            pass

    def run_cypher(self, query: str, params: Dict = None) -> List[Dict[str, Any]]:
        params = params or {}
        with self.driver.session() as session:
            result = session.run(query, **params)
            return [record.data() for record in result]

    def get_entity_candidates(self, name: str, limit: int = 5):
        q = """
        MATCH (n)
        WHERE coalesce(n.name, '') = $name
        RETURN elementId(n) AS id, n.name AS name, labels(n) AS labels
        """
        return self.run_cypher(q, {"name": name, "limit": limit})

    def get_neighbors(self, node_id: int, limit: int = 100):
        q = """
        MATCH (n)-[r]-(m)
        WHERE elementId(n) = $id
        RETURN elementId(m) AS nid, coalesce(m.name, '') AS name, type(r) AS rel, r.condition AS condition, elementId(r) AS rid
        LIMIT $limit
        """
        return self.run_cypher(q, {"id": node_id, "limit": limit})

    def execute_subgraph_query(self, node_ids: List[int]):
        q = """
        MATCH (n)-[r]-(m)
        WHERE elementId(n) IN $node_ids AND elementId(m) IN $node_ids
        RETURN elementId(n) AS src, type(r) AS rel, elementId(m) AS dst
        """
        return self.run_cypher(q, {"node_ids": node_ids})