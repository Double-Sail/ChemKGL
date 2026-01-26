import re
import ast

class Retrieval_judger:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def judger(self, query: str) -> bool:
        return True

        input_text = f"请根据你自身的能力，以及以下的问题，判断是否需要额外的检索信息来生成完整的答案。只返回[需要]或[不需要]。\n\n问题: \"{query}\""

        messages = [
            {"role": "user", "content": input_text}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if '不需要' in response:
            return False
        else:
            return True

    def extract_entity(self, query):
        input_text = f"**Task**: Extract **reactants** (substrates) and **products** (target substance) from the given chemical synthesis question. " \
                     f"Ignore supplementary reagents, conditions, or procedural steps. \n" \
                     f"**Output Format**:{{'reactants': [substrates], 'products': [target]}}\n" \
                     f"**Example**:**Input**: 'The goal is to produce the target substance ['A'] using substrates ['B', 'C', 'D']." \
                     f"Determine the required supplementary reagents...'**Output**: {{'reactants': ['B', 'C', 'D'], 'products': ['A']}}\n" \
                     f"*Instructions**:1. Identify lists explicitly marked as 'substrates' and 'target substance' in the input.2. Exclude any reagents, " \
                     f"solvents, or reaction conditions (e.g., 'required supplementary reagents'). 3. Follow the format strictly; no additional text or explanations.\n" \
                     f"Question:'{query}'"
        # input_text = f"Please extract the reactants and products from the questions I have raised. No need to answer questions, just perform extraction work, " \
        #              f"and output the results in the form of {{Catalyst:[],Product:[]}}.\n\nQuestion:'{query}'"

        messages = [
            {"role": "user", "content": input_text}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 设置填充标记 ID
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

        # 生成输入和注意力掩码
        model_inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, pad_to_multiple_of=8)
        model_inputs = model_inputs.to(self.model.device)

        # model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # generated_ids = self.model.generate(
        #     model_inputs.input_ids,
        #     max_new_tokens=1024
        # )
        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,  # 传递注意力掩码
            max_new_tokens=1024,
            pad_token_id=pad_token_id  # 传递填充标记 ID
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def extract_entity_by_re(self, query,loaddata_tag):
        if loaddata_tag == 'ome':
            reactants_str = '**Reactants (SMILES)**: {'
            reactants_start = query.find(reactants_str) + len(reactants_str)
            reactants_end = query.find('}',reactants_start)
            reactants = eval(query[reactants_start:reactants_end])

            products_str = '**Products (SMILES)**: {'
            products_start = query.find(products_str) + len(products_str)
            products_end = query.find('}',products_start)
            products = eval(query[products_start:products_end])

            assert reactants and products
            return list(set(reactants)),list(set(products))
        else:
            # 提取目标物质
            matches = re.findall(r"\['(.*?)']", query)

            if query.strip().startswith('Given substrates'):
                substrates = matches[0]
                products = matches[1]
            else:
                substrates = matches[1]
                products = matches[0]

            substrates_str = substrates.replace("'", "").replace(" ", "").replace("\\\\", "\\")
            products_str = products.replace("'", "").replace(" ", "").replace("\\\\", "\\")
            return substrates_str.split(','), products_str.split(',')
