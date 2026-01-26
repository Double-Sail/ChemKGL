from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset
import json
import os

# PATH = f'{os.getcwd()}/dataset/mydataset'
PATH = f'dataset/mydataset'
ome_prompt_path = 'oMeBench-main/prompts/default.txt'

class MyDataset(Dataset):
    def __init__(self,load_data_tag : str):
        super().__init__()

        self.load_data_tag = load_data_tag
        if self.load_data_tag == 'train':
            self.train_datas = torch.load(f'{PATH}/train_data_20.pt', weights_only=False)
            self.test_datas = torch.load(f'{PATH}/test_data_120.pt', weights_only=False)
        elif self.load_data_tag == 'ome':
            # with open('/cogbrain_data/fanyang58/MyRAG/oMeBench-main/data/oMe_Gold.json', "r") as f:
            #     data = json.load(f)
            self.train_datas = pd.read_json('oMeBench-main/data/oMe_Silver.jsonl',lines=True)
            self.test_datas = pd.read_json('oMeBench-main/data/oMe_Gold.json')
        else:
            self.train_datas = pd.read_json(f'{PATH}/train_data_top20.json', lines=True)
            self.test_datas = pd.read_json(f'{PATH}/test_data_120.json', lines=True)

        # 设置初始prompt
        if self.load_data_tag == 'ome':
            with open('oMeBench-main/prompts/default.txt','r') as f:
                self.prompt = f.read()
        else:
            self.prompt = """You are a professional organic chemist. Please provide a reaction step design based on the synthesis goal I have proposed. 

Question: [Specific issues.]

Please strictly follow the answer template below to generate answers。The answer template is as follows:
The required substrates are "XXX", "XXX", and "XXX", involving a total of [n] reaction steps.
Step 1: Using [substrate 1] and [substrate 2] as substrates, [xxx] as solvent, [xxx] as reagent, and [xxx] as catalyst, under [temperature, reaction time, pressure, pH, etc.] conditions, the final product [xxx] is generated, and the reaction equation is a chemical reaction equation in the form of [smiles].
Step 2: If this step is divided into multiple stages, it needs to be answered by stage. The first stage: [detailed reaction conditions], the second stage: [detailed reaction conditions], generates the final product [xxx], and the reaction equation is a chemical reaction equation in the form of [smiles].
……
Step n:

Be careful:
1. If there are no corresponding conditions, do not fabricate them. If there are other reaction conditions, they can also be supplemented.
2. Please output a paragraph for each step according to the natural language description, without dividing it into smaller parts.
        """
        self.graph = None
        self.graph_type = 'Chemical Graph'

    def get_train_eval_test_data_train(self):
        train_datas = []
        eval_datas = []
        test_datas = []

        for train_data in tqdm(self.train_datas, total=len(self.train_datas), desc='🚀 训练集读取', unit='entry'):
            id = train_data['id']
            question = train_data["question"]
            label = train_data['label']
            desc = train_data['desc']
            if desc == '':
                continue

            graph = train_data['graph']

            train_datas.append({'id': id, 'label': label, 'question': question, 'desc': desc, 'graph': graph})

        for test_data in tqdm(self.test_datas, total=len(self.test_datas), desc='🚀 测试集读取', unit='entry'):
            id = test_data['id']
            question = test_data["question"]
            label = test_data['label']
            desc = test_data['desc']
            # desc = '\n' + desc + '\n' + 'The above is the knowledge graph content related to the question. Please carefully analyze the search content I provided and refer to it to provide your answer. Be careful to avoid using the following statements:\n-According to the provided information\n-Search content display\n-Mentioned in the knowledge base\n-Any citation format that may expose the source of information'
            if desc == '':
                continue

            graph = test_data['graph']

            test_datas.append({'id': id, 'label': label, 'question': question, 'desc': desc, 'graph': graph})
            eval_datas = test_datas

        return train_datas, eval_datas, test_datas

    def get_train_eval_test_data_rag(self):
        train_datas = []
        eval_datas = []
        test_datas = []

        if self.load_data_tag == 'ome':
            for index, train_data in tqdm(self.train_datas.iterrows(), total=len(self.train_datas), desc='🚀 训练集读取', unit='entry'):
                reaction_id = train_data['reaction_id']
                level = train_data['level']
                name = train_data['name']
                reactants_smiles = train_data['reactants_smiles']
                products_smiles = train_data['products_smiles']

                if train_data['conditions']:
                    conditions = train_data['conditions']
                else:
                    conditions = 'Base'

                mechanism = train_data['mechanism']
                mechanism_step_nums = train_data['mechanism_step_nums']

                question = self.prompt.replace("{ reactants_smiles }", json.dumps(reactants_smiles)) \
                            .replace("{ products_smiles }", json.dumps(products_smiles)) \
                            .replace("{ conditions }", conditions)
                
                train_datas.append({'id': reaction_id, 'label': mechanism, 'question': question, 'steps': mechanism_step_nums})

            for index, train_data in tqdm(self.test_datas.iterrows(), total=len(self.test_datas), desc='🚀 测试集读取', unit='entry'):
                reaction_id = train_data['reaction_id']
                level = train_data['level']
                name = train_data['name']
                reactants_smiles = train_data['reactants_smiles']
                products_smiles = train_data['products_smiles']

                if train_data['conditions']:
                    conditions = train_data['conditions']
                else:
                    conditions = 'Base'

                mechanism = train_data['mechanism']
                mechanism_step_nums = train_data['mechanism_step_nums']

                question = self.prompt.replace("{ reactants_smiles }", json.dumps(reactants_smiles)) \
                            .replace("{ products_smiles }", json.dumps(products_smiles)) \
                            .replace("{ conditions }", conditions)
                
                test_datas.append({'id': reaction_id, 'label': mechanism, 'question': question, 'steps': mechanism_step_nums})
            eval_datas = test_datas
        else:
            for index, train_data in tqdm(self.train_datas.iterrows(), total=len(self.train_datas), desc='🚀 训练集读取', unit='entry'):
                question = self.prompt.replace('[Specific issues.]', train_data["question"])
                label = train_data['answer']
                steps = train_data['steps']

                train_datas.append({'id': index, 'label': label, 'question': question, 'steps': steps})

            id = 0
            for index, test_data in tqdm(self.test_datas.iterrows(), total=len(self.test_datas), desc='🚀 测试集读取', unit='entry'):
                id = test_data['id']
                question = test_data["question"]
                label = test_data['label']
                desc = test_data['desc']
                steps = test_data['steps']

                test_datas.append({'id': id, 'label': label, 'question': question, 'desc': desc, 'steps': steps})
            eval_datas = test_datas

        return train_datas, eval_datas, test_datas

    def get_relation_dict(self):
        relation_name2ids = pd.read_csv(f'{PATH}/relation_name2id.csv')
        relation_dict_name2ids = {}

        for _, relation_name2id in relation_name2ids.iterrows():
            relation_dict_name2ids[relation_name2id[0]] = relation_name2id[1]

        relation_dict_id2names = {v: k for k, v in relation_dict_name2ids.items()}
        return relation_dict_name2ids, relation_dict_id2names


if __name__ == '__main__':
    dataset = MyDataset()
