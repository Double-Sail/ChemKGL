# src/utils.py
import json
import re
from typing import List, Dict, Any
import prompts

# def parse_json_safe(text: str):
#     try:
#         return json.loads(text)
#     except Exception:
#         # try to extract first JSON array/object
#         m = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
#         if m:
#             try:
#                 return json.loads(m.group(1))
#             except Exception:
#                 return None
#         return None

def parse_json_safe(text: str):
    # 1. 优先匹配 ```json ... ```
    m = re.search(
        r"```json\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```",
        text
    )
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 2. fallback：匹配第一个 JSON array / object（非贪婪）
    m = re.search(
        r"(\[[\s\S]*?\]|\{[\s\S]*?\})",
        text
    )
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None

    return None

def merge_subgraphs(subgraphs: List[Any], strategy: str = 'most_visited'):
    nodes = set()
    edges = set()
    for s in subgraphs:
        nodes.update(s.nodes)
        edges.update(s.edges)
    return {'nodes': list(nodes), 'edges': [list(e) for e in edges]}

def get_subgraphs_desc(subgraphs):
    required_substrates = []
    relations=[]
    for s in subgraphs:
        required_substrates = s.frontier_name[:-1]
        relations = [i[1] for i in s.edges]
    return prompts.answer_prompt(required_substrates,relations)