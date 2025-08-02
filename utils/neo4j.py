from py2neo import Graph, Node, NodeMatcher, RelationshipMatcher, ConnectionUnavailable, Relationship
import os
import pandas as pd
from tqdm import tqdm
import re
from datetime import datetime


# def csv2neo4j(node_file, relation_file):
#     db = connect_neo4j("http://172.31.169.49:7474", "neo4j", "yf939924")
#
#     node_id_name = {}
#     # 创建节点
#     nodes = pd.read_csv(node_file)
#     for node in tqdm(nodes.iterrows(), total=len(nodes)):
#         name = node[1][1]
#         id = node[1][0]
#         node = Node('entity', name=name, id=id)
#         db.create(node)
#         node_id_name[id] = name
#     print('节点创建完毕！！！！')
#
#     # 创建关系
#     relations = pd.read_csv(relation_file)
#     for node in tqdm(relations.iterrows(), total=len(relations)):
#         head_id, relation, tail_id = node[1][0].split('|')
#
#         head_name = node_id_name[int(head_id)]
#         tail_name = node_id_name[int(tail_id)]
#
#         id = node[1][1]
#         node = Relationship(head_name, relation, tail_name)
#         db.create(node)
#     print('关系创建完毕！！！！')
#     pass

def csv2neo4j(node_file, relation_file):
    # 连接到 Neo4j
    graph = Graph("http://172.31.169.49:7474", auth=("neo4j", "yf939924"))

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
        head_id = str(head_id)
        tail_id = str(tail_id)

        if head_id in nodes_dict and tail_id in nodes_dict:
            head_node = nodes_dict[head_id]
            tail_node = nodes_dict[tail_id]

            # 创建关系
            rel_between_nodes = Relationship(head_node, relation, tail_node, rel_id=rel[1])
            graph.create(rel_between_nodes)
        else:
            print(f"未能找到头或尾节点: {head_id} 或 {tail_id}")
    print('关系创建完毕！！！！')


def three_couple_2_csv(file_root_path, dataset_name):
    """
    生成两个csv文件，node文件：包含node名称和id，关系文件：id1 关系 id2
    """
    nodes_name2id = {}
    relation_id2name2id = {}
    # relation_id2id2id = {}
    relation_name2id = {}

    if dataset_name == 'grag':
        csv_dataset = pd.read_csv(f'{file_root_path}/train_dev.tsv', sep='\t')
        for i, row in tqdm(csv_dataset.iterrows(), total=len(csv_dataset)):
            graph = row['graph']
            triplets = re.findall(r'\((.*?)\)', graph)
            for tri in triplets:
                head, edeg_attr, tail = tri.split(';')
                head = head.lower().strip()
                tail = tail.lower().strip()

                if edeg_attr in relation_name2id:
                    edeg_attr_id = relation_name2id[edeg_attr]
                else:
                    edeg_attr_id = len(relation_name2id)
                    relation_name2id[edeg_attr] = edeg_attr_id

                if head not in nodes_name2id:
                    nodes_name2id[head] = len(nodes_name2id)
                if tail not in nodes_name2id:
                    nodes_name2id[tail] = len(nodes_name2id)

                relation_id2name2id_key = f'{nodes_name2id[head]}|{edeg_attr}|{nodes_name2id[tail]}'
                # relation_id2id2id_key = f'{nodes_name2id[head]}|{edeg_attr_id}|{nodes_name2id[tail]}'
                if relation_id2name2id_key not in relation_id2name2id:
                    relation_id2name2id[relation_id2name2id_key] = relation_name2id[edeg_attr]
                # if relation_id2id2id_key not in relation_id2id2id:
                #     relation_id2id2id[relation_id2id2id_key] = len(relation_id2id2id)

        nodes_name2id = pd.DataFrame(nodes_name2id.items(), columns=['node_name', 'node_id'])
        relation_id2name2id = pd.DataFrame(relation_id2name2id.items(), columns=['relation', 'relation_id'])
        # relation_id2id2id = pd.DataFrame(relation_id2id2id.items(), columns=['relation', 'relation_id'])
        relation_name2id = pd.DataFrame(relation_name2id.items(), columns=['relation_name', 'relation_id'])
        # revserse_relation_name2id = pd.DataFrame(revserse_relation_name2id.items(), columns=['relation_name', 'relation_id'])

        nodes_name2id.to_csv(f'{file_root_path}/nodes.csv', index=False, columns=['node_id', 'node_name'])
        relation_id2name2id.to_csv(f'{file_root_path}/relation_id2name2id.csv', index=False, columns=['relation', 'relation_id'])
        # relation_id2id2id.to_csv(f'{file_root_path}/relation_id2id2id.csv', index=False, columns=['relation', 'relation_id'])
        relation_name2id.to_csv(f'{file_root_path}/relation_name2id.csv', index=False, columns=['relation_name', 'relation_id'])
        # revserse_relation_name2id.to_csv(f'{file_root_path}/revserse_relation_name2id.csv', index=False, columns=['relation_id', 'relation_name'])

    print('csv文件生成完成！！！')


def connect_neo4j(url, user, password):
    return Graph(url, auth=(user, password))


if __name__ == '__main__':
    dataset_name = 'grag'
    three_couple_2_csv('/code/KGLLM2.0/MyRAG/dataset/expla_graphs', dataset_name)
    csv2neo4j('/code/KGLLM2.0/MyRAG/dataset/expla_graphs/nodes.csv', '/code/KGLLM2.0/MyRAG/dataset/expla_graphs/relation_id2name2id.csv')

    graph = Graph("http://172.31.169.49:7474", auth=("neo4j", "yf939924"))
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
    pass

