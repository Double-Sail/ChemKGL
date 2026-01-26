import json
from run import load_prompt_template,build_prompt,ROOT,PROMPT_PATH,INPUT_PATH
from tqdm import tqdm

if __name__ == "__main__":
    prompt_template = load_prompt_template(PROMPT_PATH)

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)
    
    gold_data = ROOT

    for idx, reaction in tqdm(enumerate(data), total=len(data), desc=f"Generating with"):
            rxn_id = reaction.get("reaction_id", f"rxn_{idx}")
            prompt = build_prompt(prompt_template, reaction['reactants_smiles'], reaction['products_smiles'], reaction['conditions'])
