"""
依据gold数据集生成知识图谱，为了避免节点太少，将silver和template的数据也参与图谱的生成
"""

import py2neo
import json
from pathlib import Path
from tqdm import tqdm
from run import parse_args,ROOT,PROMPT_PATH,INPUT_PATH,GOLD_PATH
import pandas as pd
import os
from py2neo import Graph, Node, Relationship
from datetime import datetime

def triple2csv(file_root_path,input_file_path):
    """
    生成两个csv文件，node文件：包含node名称和id，关系文件：id1 关系 id2
    """
    nodes_name2id = {}
    relation_id2name2id = {}
    relation_name2id = {}
    csv_dataset = pd.read_csv(f'{file_root_path}/{input_file_path}', sep='\t')

    for i, row in tqdm(csv_dataset.iterrows(), total=len(csv_dataset)):
        tri = row[0]
        head, relation, relation_description, tail , history = tri.split('|')
        head = head.strip()
        tail = tail.strip()

        if relation_description in relation_name2id:
            edeg_attr_id = relation_name2id[relation_description]
        else:
            edeg_attr_id = len(relation_name2id)
            relation_name2id[relation_description] = edeg_attr_id

        if head not in nodes_name2id:
            nodes_name2id[head] = len(nodes_name2id)
        if tail not in nodes_name2id:
            nodes_name2id[tail] = len(nodes_name2id)
        relation_id2name2id_key = f'{nodes_name2id[head]}|{relation_description}|{nodes_name2id[tail]}'
        if relation_id2name2id_key not in relation_id2name2id:
            hset = set()
            for a in history.split(','):
                if not isinstance(a, str) or a == ' ' or a == '':
                    continue
                hset.add(a)
            relation_id2name2id[relation_id2name2id_key] = {'relation_id': edeg_attr_id, 'history': hset}
        else:
            for a in history.split(','):
                if not isinstance(a, str) or a == ' ' or a == '':
                    continue
                relation_id2name2id[relation_id2name2id_key]['history'].add(a)

    nodes_name2id = pd.DataFrame(nodes_name2id.items(), columns=['node_name', 'node_id'])
    nodes_name2id.to_csv(f'{file_root_path}/nodes.csv', index=False, columns=['node_id', 'node_name'])

    rows = []
    for outer_key, inner_dict in relation_id2name2id.items():
        history = ','.join(inner_dict['history'])
        if not isinstance(history, str) or history is None or history == '':
            history = ''
        else:
            history = f'This reaction has been used to generate {history}'
        # if history == 'This reaction has been used to generate ':
        #     print(1)
        row = {
            'relation': outer_key,
            'relation_id': inner_dict['relation_id'],
            'history': history
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f'{file_root_path}/relation_id2name2id.csv', index=False)

    print('csv文件生成完成！！！')


def csv2neo4j(node_file, relation_file):
    # 连接到 Neo4j
    graph = Graph("ip_address", auth=("admin", "passwd"))
    meta_node = Node(
        'Meta',
        id='meta-001',
        title='myrag meta node',
        text='store some meta info',
        # python获取当前时间
        timestamp=str(datetime.now()),
        version=0,
        status='active',
    )
    graph.create(meta_node)

    nodes_dict = {}

    # 创建节点
    nodes = pd.read_csv(node_file)
    for _, node in tqdm(nodes.iterrows(), total=len(nodes)):
        name = str(node[1])
        id = str(node[0])  # 确保ID是字符串形式
        neo_node = Node('entity', name=name, id=id)
        graph.create(neo_node)
        nodes_dict[id] = neo_node  # 存储创建的节点对象以便后续关系创建使用
    print('节点创建完毕！！！！')

    # 创建关系
    relations = pd.read_csv(relation_file)
    for _, rel in tqdm(relations.iterrows(), total=len(relations)):
        head_id, relation, tail_id = rel[0].split('|')
        history = rel[2]
        if not isinstance(history, str):
            history = ''

        head_id = str(head_id)
        tail_id = str(tail_id)

        if head_id in nodes_dict and tail_id in nodes_dict:
            # 原始创建
            head_node = nodes_dict[head_id]
            tail_node = nodes_dict[tail_id]

            # 创建关系
            rel_between_nodes = Relationship(head_node, 'PRODUCT', tail_node, condition=str(relation), history=history, rel_id=rel[1])
            graph.create(rel_between_nodes)
        else:
            print(f"未能找到头或尾节点: {head_id} 或 {tail_id}")

    print('关系创建完毕！！！！')

def main():
    # Set output paths
    output_path = ROOT / "graphdata" / f"KG_triple.csv"

    if not os.path.exists(output_path):
        output_path.parent.mkdir(exist_ok=True)
        with open(GOLD_PATH, "r") as f:
            gold_data = json.load(f)
        with open(TEMPLATE_PATH, "r") as f:
            template_data = json.load(f)

        silver_data = []
        with open(SILVER_PATH, "r") as json_files:
            alllines = json_files.readlines()
            for line in alllines:
                silver_data.append(json.loads(line))

        triples = []
        for data_name in [gold_data,template_data,silver_data]:
            for idx, reaction in tqdm(enumerate(data_name), total=len(data_name), desc=f"Generating {data_name} csv"):
                reactants_smiles = reaction.get('reactants_smiles')
                reactants_smiles = set(reactants_smiles)
                mechanism = reaction.get('mechanism')
                final_products = reaction.get('products_smiles')

                for final_product in final_products:
                    if final_product not in mechanism[-1]['intermediate_smiles']:
                        mechanism[-1]['intermediate_smiles'] = mechanism[-1]['intermediate_smiles'] + "." + final_product

                for step_dict in mechanism:
                    for reactants_smile in reactants_smiles:
                        if 'subtype' in step_dict:
                            graph_condition = str({"type":step_dict['type'],'subtype':step_dict['subtype'] ,'intermediate_smiles':step_dict['intermediate_smiles']})
                        else:
                            graph_condition = str({"type":step_dict['type'],'subtype':'' ,'intermediate_smiles':step_dict['intermediate_smiles']})

                        intermediate_smiles = step_dict['intermediate_smiles'].split('.')
                        assert ',' not in intermediate_smiles
                        for intermediate_smile in intermediate_smiles:
                            products_smile = intermediate_smile

                            if products_smile in final_products or step_dict['step'] == len(mechanism):
                                triple_str = reactants_smile + "|" + 'PRODUCT' + "|" + graph_condition + "|" + products_smile + "|"
                            else:
                                triple_str = reactants_smile + "|" + 'PRODUCT' + "|" + graph_condition + "|" + products_smile + "|" + ",".join(final_products)
                            triples.append(triple_str)

                    reactants_smiles = intermediate_smiles

        triples = pd.DataFrame(triples)
        triples.to_csv(output_path, index=False)
    
    csv_root_path = 'oMeBench-main/graphdata'
    if not os.path.exists(f'{csv_root_path}/relation_id2name2id.csv'):
        triple2csv(csv_root_path,f'KG_triple.csv')
    csv2neo4j(f'{csv_root_path}/nodes.csv', f'{csv_root_path}/relation_id2name2id.csv')

if __name__ == "__main__":
    TEMPLATE_PATH = ROOT/ "data" / "oMe_Template.json"
    SILVER_PATH = ROOT/ "data" / "oMe_Silver.jsonl"
    main()