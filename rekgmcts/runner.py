# src/runner.py
from neo4j_utils import Neo4jUtils
from model import DummyLLM, LocalHFLLM
from mcts import MCTS, SubgraphState
import utils
import re

class Runner:
    def __init__(self, cfg: dict, demo_mode: bool = False):
        self.cfg = cfg
        self.demo_mode = demo_mode

        # -----------------------------
        # Neo4j or Demo Graph
        # -----------------------------
        if demo_mode:
            self.neo = None
        else:
            self.neo = Neo4jUtils(
                cfg['neo4j']['uri'],
                cfg['neo4j']['user'],
                cfg['neo4j']['password']
            )

        # -----------------------------
        # Choose LLM
        # -----------------------------
        model_cfg = cfg.get('model', {})
        model_type = model_cfg.get('type', 'dummy')

        if model_type == 'local' and not demo_mode:
            # 使用本地 LLaMA / Qwen
            self.llm = LocalHFLLM(
                model_path=model_cfg['model_path'],
                device=model_cfg.get('device', 'cuda'),
                max_new_tokens=model_cfg.get('max_new_tokens', 256),
            )
        else:
            # Dummy baseline / demo fallback
            self.llm = DummyLLM()

        # -----------------------------
        # MCTS
        # -----------------------------
        self.mcts = MCTS(
            self.neo if self.neo is not None else self._demo_neo(),
            self.llm,
            cfg
        )

    # ======================================================
    # Demo Neo4j (in-memory graph)
    # ======================================================
    def _demo_neo(self):
        class DemoNeo:
            def __init__(self):
                self.nodes = {
                    1: "Albert Einstein",
                    2: "Max Planck",
                    3: "Princeton University",
                    4: "Physics"
                }
                self.edges = {
                    1: [(2, "influenced_by"), (3, "affiliated_with"), (4, "field_of")],
                    2: [(1, "influenced"), (4, "field_of")],
                    3: [(1, "affiliated_member")],
                    4: [(1, "field_of"), (2, "field_of")]
                }

            def get_entity_candidates(self, name, limit=5):
                res = []
                for nid, nm in self.nodes.items():
                    if name.lower() in nm.lower():
                        res.append({"id": nid, "name": nm})
                if not res:
                    res.append({"id": 1, "name": "Albert Einstein"})
                return res[:limit]

            def get_neighbors(self, node_id, limit=100):
                res = []
                for (dst, rel) in self.edges.get(node_id, []):
                    res.append({
                        "nid": dst,
                        "name": self.nodes.get(dst, ""),
                        "rel": rel
                    })
                return res

        return DemoNeo()

    # ======================================================
    # Run one question
    # ======================================================
    def run_question(self, question: str,substrates, seed_entity_id: int = None):
        self.cfg['question'] = question
        neo = self._demo_neo() if self.demo_mode else self.neo

        if seed_entity_id is None:
            cand = neo.get_entity_candidates(substrates[0], limit=1)
            if not cand:
                raise RuntimeError("No candidate entities found for the question.")
            seed_entity_id = cand[0]['id']
            seed_entity_name = cand[0]['name']

        root_state = SubgraphState(
            nodes_id={seed_entity_id},
            nodes_name={seed_entity_name},
            edges=set(),
            frontier_id=[seed_entity_id],
            frontier_name=[seed_entity_name],
        )

        root = self.mcts.search(root_state)

        leaves = self._collect_leaves(root)
        leaves_sorted = sorted(leaves, key=lambda n: n.visits, reverse=True)

        top_k = self.cfg.get('output', {}).get('top_k_paths')
        subgraphs = [leaf.state for leaf in leaves_sorted[:top_k]]

        return utils.get_subgraphs_desc(subgraphs)

    def _collect_leaves(self, node):
        if not node.children:
            return [node]
        leaves = []
        for c in node.children:
            leaves.extend(self._collect_leaves(c))
        return leaves
