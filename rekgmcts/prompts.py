# src/prompts.py
from typing import List, Dict

def policy_prompt(question: str, subgraph_summary: str, candidates: List[Dict]):
    cand_text = "\n".join([f"{i+1}. rel={c.get('condition')}, ent={c.get('name')} (nid={c.get('nid')})" for i, c in enumerate(candidates)])
    return f"""
Question: {question}

Current subgraph: {subgraph_summary}

Candidate expansions:
{cand_text}

Return a JSON array like:
[{{"ent":"...", "nid":123, "score":0.32}}, ...]
where score is between 0 and 1 representing how relevant each expansion is to the question.
Only return valid JSON.
""".strip()

def value_prompt(question: str, subgraph_summary: str) -> str:
    return f"""
Question: {question}

Subgraph summary:
{subgraph_summary}

Return a single number between 0 and 1 representing how relevant this subgraph is to the question.
Only return the number.
""".strip()

def answer_prompt(required_substrates,relations):
    desc_correct = f'Please refer to the following information to answer,be careful not to reveal in your answer that you referenced the information I provided.\n' \
                    f'The required substrates are {" and ".join(required_substrates)}, involving a total of {len(relations)} reaction steps.\n'
    for idx, reaction_path in enumerate(relations):
        desc_correct += f'Step{idx + 1}:{reaction_path}\n'
    return desc_correct