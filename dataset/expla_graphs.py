import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import os

PATH = f'{os.getcwd()}/dataset/expla_graphs'


class ExplaGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.text = pd.read_csv(f'{PATH}/train_dev.tsv', sep='\t')
        self.prompt = 'Question: Do argument 1 and argument 2 support or counter each other? Answer in one word in the form of \'support\' or \'counter\'.\n\nAnswer:'
        self.graph = None
        self.graph_type = 'Explanation Graph'

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):

        text = self.text.iloc[index]
        # graph = torch.load(f'{PATH}/graphs/{index}.pt')
        question = f'Argument 1: {text.arg1}\nArgument 2: {text.arg2}\n{self.prompt}'
        # nodes = pd.read_csv(f'{PATH}/nodes/{index}.csv')
        # edges = pd.read_csv(f'{PATH}/edges/{index}.csv')
        # desc = nodes.to_csv(index=False)+'\n'+edges.to_csv(index=False)

        return {
            'id': index,
            'label': text['label'],
            # 'desc': desc,
            # 'graph': graph,
            'question': question,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{PATH}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open(f'{PATH}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}

    # def get_triplets_dict(self):
    #     relation_id2id2ids = pd.read_csv(f'{PATH}/relation_id2id2id.csv')
    #     triplets_dict = {}
    #     for _, relation_id2id2id in relation_id2id2ids.iterrows():
    #         relation = relation_id2id2id[0]
    #         key = relation.split('|')[0]
    #         triplets_dict[key] = relation
    #     return triplets_dict

    def get_relation_dict(self):
        relation_name2ids = pd.read_csv(f'{PATH}/relation_name2id.csv')
        relation_dict_name2ids = {}

        for _, relation_name2id in relation_name2ids.iterrows():
            relation_dict_name2ids[relation_name2id[0]] = relation_name2id[1]

        relation_dict_id2names = {v: k for k, v in relation_dict_name2ids.items()}
        return relation_dict_name2ids, relation_dict_id2names


if __name__ == '__main__':
    dataset = ExplaGraphsDataset()
