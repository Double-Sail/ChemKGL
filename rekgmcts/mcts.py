# src/mcts.py
from typing import Any, Dict, List, Set, Tuple
import math
import random
import prompts

class SubgraphState:
    def __init__(self, nodes_id: Set[int],nodes_name: Set[str], edges: Set[Tuple[int, str, int]], frontier_id: List[int],frontier_name: List[str]):
        self.nodes_id = set(nodes_id)
        self.nodes_name = set(nodes_name)
        self.edges = set(edges)
        self.frontier_id = list(frontier_id)
        self.frontier_name = list(frontier_name)

    def copy_and_add(self, src_id: int,src_name, rel: str, dst_id: int,dst_name):
        new_nodes_id = set(self.nodes_id)
        new_nodes_name = set(self.nodes_name)
        new_edges = set(self.edges)

        new_nodes_id.add(dst_id)
        new_nodes_name.add(dst_name)

        new_edges.add((src_name, rel, dst_name))

        new_frontier_id = list(self.frontier_id)
        new_frontier_name = list(self.frontier_name)

        if dst_id not in new_frontier_id:
            new_frontier_id.append(dst_id)
            new_frontier_name.append(dst_name)
        return SubgraphState(new_nodes_id,new_nodes_name, new_edges, new_frontier_id,new_frontier_name)

class MCTSNode:
    def __init__(self, state: SubgraphState, parent=None, prior=1.0):
        self.state = state
        self.parent = parent
        self.depth = 0 if parent is None else parent.depth + 1
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.prior = prior
        self.action = None  # (src, rel, dst)

    def is_fully_expanded(self, max_width: int,max_depth: int):
        # 1. 深度限制：达到最大深度，禁止继续扩展
        if max_depth is not None and self.depth >= max_depth:
            return True

        # 2. 没有 frontier，本身就无法扩展
        if not self.state.frontier_id:
            return True

        # 3. 宽度限制
        return len(self.children) >= max_width

    def best_ucb_child(self, c_puct: float):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            q = 0.0 if child.visits == 0 else child.value / child.visits
            u = c_puct * child.prior * math.sqrt(self.visits + 1) / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

class MCTS:
    def __init__(self, neo4j_utils, llm, cfg: Dict):
        self.neo = neo4j_utils
        self.llm = llm
        self.cfg = cfg

    def search(self, root_state: SubgraphState):
        root = MCTSNode(root_state)
        iters = int(self.cfg['mcts'].get('iterations', 100))
        max_width = self.cfg['mcts']['max_width']
        max_depth = self.cfg['mcts']['max_depth']

        for it in range(iters):
            # ⭐ 全局终止条件
            if max_depth is not None and self._all_leaves_reached_max_depth(root, max_depth):
                break

            node = root
            path = [node]

            # Selection
            while node.children and node.is_fully_expanded(max_width,max_depth):
                node = node.best_ucb_child(self.cfg['mcts']['exploration_constant'])
                path.append(node)

            # Expansion         
            if not node.is_fully_expanded(max_width,max_depth):
                if not node.state.frontier_id:
                    val = self.evaluate(node.state)
                    self.backpropagate(path, val)
                    continue

                src_id = random.choice(node.state.frontier_id)
                src_name = random.choice(node.state.frontier_name)
                neighbors = []
                try:
                    neighbors = self.neo.get_neighbors(src_id, limit=self.cfg['mcts'].get('neighbor_limit', 20))
                except Exception:
                    neighbors = []

                candidates = []
                id2condtion = {}
                # 避免condition过长，导致max_new_token太长，condition不参与输出，直接字典对应
                for n in neighbors:
                    candidates.append({'condition':n.get('condition', ''),'rel': n.get('rel', ''), 'name': n.get('name', ''), 'nid': n.get('nid')})
                    id2condtion[n.get('nid')] = n.get('condition', '')

                # Ask policy
                prompt = prompts.policy_prompt(self.cfg.get('question', ''), self._summarize(node.state), candidates)
                priors = self.llm.score_policy(prompt)
                if not priors:
                    # fallback uniform
                    if candidates:
                        uniform = 1.0 / len(candidates)
                        priors = [{'name': c['name'], 'nid': c['nid'], 'score': uniform} for c in candidates]
                    else:
                        priors = []

                cheak_priors = []
                for p in priors:
                    id = p.get('nid')
                    if id in id2condtion:
                        cheak_priors.append(p)
                priors = cheak_priors

                top_k = int(self.cfg['mcts'].get('top_k_expansions', 3))
                priors_sorted = sorted(priors, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
                for p in priors_sorted:
                    condition = id2condtion[p.get('nid')]
                    child_state = node.state.copy_and_add(src_id,src_name, condition, p.get('nid', -1),p.get('ent', ''))
                    child = MCTSNode(child_state, parent=node, prior=p.get('score', 0.01))
                    child.action = (src_name, condition, p.get('ent', -1))
                    node.children.append(child)

                if node.children:
                    node = random.choice(node.children)
                    path.append(node)

            # Simulation / Evaluation
            value = self.evaluate(node.state)
            self.backpropagate(path, value)

        return root

    def evaluate(self, state: SubgraphState) -> float:
        prompt = prompts.value_prompt(self.cfg.get('question', ''), self._summarize(state))
        v = self.llm.score_value(prompt)
        return float(max(0.0, min(1.0, v)))

    def backpropagate(self, path: List[MCTSNode], value: float):
        for node in reversed(path):
            node.visits += 1
            node.value += value

    def _summarize(self, state: SubgraphState) -> str:
        # nodes = list(state.nodes_id)[:10]
        nodes = list(state.nodes_name)[:10]
        edges = list(state.edges)[:10]
        return f"nodes={nodes}, edges={edges}"
    
    def _all_leaves_reached_max_depth(self, root: MCTSNode, max_depth: int) -> bool:
        stack = [root]
        while stack:
            node = stack.pop()
            if not node.children:  # 叶子节点
                if node.depth < max_depth:
                    return False
            else:
                stack.extend(node.children)
        return True

