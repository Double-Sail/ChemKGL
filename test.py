import json
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd
import os
import zipfile
import json
import pandas as pd
import math
from tqdm import tqdm
from py2neo import Graph, Node, Relationship
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import copy
import random
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import requests
from urllib.parse import quote
from datetime import datetime
import re


# with open('D:\Code\MyRAG\dataset\mydataset\修改测试集\J-3695_40条问题_export_all_硅基-qwq-32b.json','r',encoding='utf-8') as jsonfiles:
#     for jsonfile in jsonfiles:
#         jsondict = json.loads(jsonfile)
#         print(1)

class LLM:
    def __init__(self):
        kwargs = {
            "max_memory": {0: '40GiB', 1: '40GiB', 2: '40GiB', 3: '40GiB'},
            "device_map": f"cuda:{1}",
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained('/cogbrain/zmliu7/Meta-Llama-3.1-8B-Instruct/', use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        self.model = AutoModelForCausalLM.from_pretrained(
            '/cogbrain/zmliu7/Meta-Llama-3.1-8B-Instruct/',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )

    def generate(self, input_text):
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
            max_new_tokens=2048,
            pad_token_id=pad_token_id  # 传递填充标记 ID
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


if __name__ == '__main__':
    # llama3对条件的文本描述
    # llm = LLM()
    #
    # while True:
    #     print('='*50)
    #     input_text = input('输入问题:')
    #     result = llm.generate(input_text)
    #     print(result)

    # 测试集的答案模板更改
    # 40条交付_test_data_2_5_modify.json
    # from dataset.mydataset.data_select_modify_generate import generate_relation_description
    # with open('/cogbrain/yjsun14/code/KGLLM2.0/MyRAG/dataset/mydataset/修改测试集/40条交付_test_data_2_5_modify.json', 'r', encoding='utf-8') as files:
    #     datas = json.load(files)
    #
    # for data in datas:
    #     print(1)

    # start_multisteps = 2
    # end_multisteps = 5
    # root_path = '/cogbrain/yjsun14/code/KGLLM2.0/MyRAG/dataset/mydataset'
    # fast_data_id = set()
    # with open(f'{root_path}/train_data_{start_multisteps}_{end_multisteps}_fast.json', 'r', encoding='utf-8') as files1:
    #     for file in files1:
    #         jsondict = json.loads(file)
    #         reactionid = jsondict['reactionid']
    #         fast_data_id.add(reactionid)
    #
    need_datas = []
    # with open(f'{root_path}/train_data_{start_multisteps}_{end_multisteps}.json', 'r', encoding='utf-8') as files2:
    #     for file in files2:
    #         jsondict = json.loads(file)
    #         reactionid = jsondict['reactionid']
    #         if reactionid in fast_data_id:
    #             need_datas.append(jsondict)
    #
    # print(len(need_datas))
    # with open(f'{root_path}/train_data_{start_multisteps}_{end_multisteps}_fast_10.json', 'w', encoding='utf-8') as f:
    #     for need_jsondict in need_datas:
    #         json.dump(need_jsondict, f, ensure_ascii=False)
    #         f.write('\n')
    # num = 0
    # dict = {'v3': 0, 'qwq': 0, 'checked': 0}
    # with open(f'{root_path}/修改测试集/J-3751-52-check_QA_export_all_hsq_gpt4o_1120_fxx_4o_1120.json', 'r', encoding='utf-8') as files1:
    #     for file in files1:
    #         jsondict = json.loads(file)
    #         model = jsondict['model']
    #         answer = jsondict['answer']
    #         ori_id = jsondict['ori_id']
    #         try:
    #             last_index1 = answer.rfind('最终得分')
    #             last_index2 = answer.find('分', last_index1 + 5)
    #             score = int(answer[last_index1 + 5:last_index2].strip())
    #             dict[model] += score
    #             print(f'{model}:{score}')
    #         except:
    #             print(model)
    #
    #     v3 = (dict['v3'] - 6) / 38
    #     qwq = dict['qwq'] / 40
    #     checked = dict['checked'] / 40
    #     print(f'v3平均分：{v3},qwq平均分：{qwq},checked平均分：{checked}')

    # import re
    #
    # text = "Given substrates ['CCCCCCCCCCCCCCCCO'] as starting materials, propose the missing reagents required for the synthesis of product ['CCCCCCCCCCCCCCCCc1ccco1']. " \
    #        "Include a complete reaction mechanism and provide reaction conditions for each stage."
    #
    # # 使用非贪婪匹配，确保精准提取方括号内的内容
    # matches = re.findall(r"\['(.*?)']", text)
    # print(matches)

    # graph = Graph("http://172.31.169.49:7474", auth=("neo4j", "yf939924"))
    # train_datas = torch.load(f'/cogbrain/yjsun14/code/KGLLM2.0/MyRAG/dataset/mydataset/train_data_20000.pt')
    # for train_data in tqdm(train_datas, total=len(train_datas), desc='🚀 训练集读取', unit='entry'):
    #     id = train_data['id']
    #     question = train_data["question"]
    #     label = train_data['label']
    #     desc = train_data['desc']
    #
    #     node_str, edge_str = desc.split('\n\n')
    #     node_list = node_str.split('\n')
    #     edge_list = edge_str.split('\n')
    #
    #     node_list = node_list[1:]
    #     edge_list = edge_list[1:-1]
    #
    #     graph = train_data['graph']
    #
    #     train_datas.append({'id': id, 'label': label, 'question': question, 'desc': desc, 'graph': graph})

    # v3_need_dicts = []
    # qwq_need_dicts = []
    # gemini_need_dicts = []
    #
    # test_datas = torch.load('/cogbrain/yjsun14/code/KGLLM2.0/MyRAG/dataset/mydataset/test_data_2000.pt')
    # for test_data in test_datas:
    #     v3_need_dicts.append({'query': test_data['question'], 'id': test_data['id'], 'model_name': 'v3'})
    #     qwq_need_dicts.append({'query': test_data['question'], 'id': test_data['id'], 'model_name': 'qwq'})
    #     gemini_need_dicts.append({'query': test_data['question'], 'id': test_data['id'], 'model_name': 'gemini'})
    #
    # with open('/cogbrain/yjsun14/code/KGLLM2.0/MyRAG/dataset/mydataset/v3回答结果.json', 'w', encoding='utf-8') as f:
    #     for need_jsondict in v3_need_dicts:
    #         json.dump(need_jsondict, f, ensure_ascii=False)
    #         f.write('\n')
    #
    # with open('/cogbrain/yjsun14/code/KGLLM2.0/MyRAG/dataset/mydataset/qwq回答结果.json', 'w', encoding='utf-8') as f:
    #     for need_jsondict in qwq_need_dicts:
    #         json.dump(need_jsondict, f, ensure_ascii=False)
    #         f.write('\n')
    #
    # with open('/cogbrain/yjsun14/code/KGLLM2.0/MyRAG/dataset/mydataset/gemini回答结果.json', 'w', encoding='utf-8') as f:
    #     for need_jsondict in gemini_need_dicts:
    #         json.dump(need_jsondict, f, ensure_ascii=False)
    #         f.write('\n')
    a = torch.load('/cogbrain/yjsun14/code/KGLLM2.0/MyRAG/dataset/mydataset/test_data_120.pt', weights_only=False)
    print(1)
