import sys
from pathlib import Path

# 获取当前脚本的绝对路径的父级目录（即model/的父目录MyRAG/）
current_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(current_dir))

import contextlib
import json
import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from torch_scatter import scatter
from gnn import load_gnn_model
from retrieval.retrieval_judger import Retrieval_judger
from prune.prune import Extract, Prune
from torch_geometric.data.data import Data
import pandas as pd
import logging
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import os
# import wandb
import gc
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.config import parse_args
from dataset import load_dataset
from utils.seed import seed_everything
from utils.collate import collate_fn
from retrieval.graph_dao import GraphDao

handlers = [
    logging.FileHandler('', mode='w'),
    logging.StreamHandler()  # 打印到终端
]
# logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
logger = logging.getLogger(__name__)
for handler in handlers:
    logger.addHandler(handler)


# from peft import (
#     LoraConfig,
#     get_peft_model,
#     prepare_model_for_kbit_training,
# )


class Sentence_Transformer(nn.Module):

    def __init__(self, pretrained_repo):
        super(Sentence_Transformer, self).__init__()
        logger.info(f"inherit model weights from {pretrained_repo}")
        self.bert_model = AutoModel.from_pretrained(pretrained_repo)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = self.mean_pooling(bert_out, attention_mask)

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


class GraphLLMProcess(torch.nn.Module):

    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        # self.relation_dict_name2ids = relation_dict_name2ids
        # self.reverse_relation_dict = relation_dict_id2names

        kwargs = {
            "max_memory": {0: '40GiB', 1: '40GiB', 2: '40GiB', 3: '40GiB'},
            "device_map": f"cuda:{args.device}",
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            logger.info("Freezing model!")
            for _, param in model.named_parameters():
                param.requires_grad = False

        self.model = model
        logger.info('Finish loading model!')

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            mlp_layers=args.alignment_mlp_layers,
            num_heads=args.gnn_num_heads,
            operator=args.distance_operator,
        ).to(self.model.device)

        # If you are using llama2-13b, replace with nn.Linear(2048, 5120) ...
        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

        # 构建检索判断器，实体检索器，关系抽取器，剪枝器
        self.retrieval_judger = Retrieval_judger(model=self.model, tokenizer=self.tokenizer)

        self.extracter = Extract(self.model, self.tokenizer, args)
        self.pruner = Prune(self.model, self.tokenizer, args)

        # 构建编码器模型
        self.embedding_model = Sentence_Transformer(args.embedding_model_path)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model_path)
        self.embedding_model.to(self.model.device)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self):
        # def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast()
        else:
            return contextlib.nullcontext()

    def encode_graphs(self, graphs):
        graphs = graphs.to(self.model.device)
        n_embeds, _ = self.graph_encoder(graphs.x,
                                         graphs.edge_index.long(),
                                         graphs.question_node,
                                         graphs.edge_attr,
                                         graphs.question_edge)

        # mean pooling
        g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')

        return g_embeds

    def forward(self, samples):
        # 根据问题进行数据库检索，并根据查询+剪枝之后的结果，得到图谱文本描述desc和子图graph
        desc_list = []
        graph_list = []

        for sample in samples['question']:
            pruned_fact_strs = []
            pruned_fact_triples_list = []

            querys = sample.split('\n')
            query = querys[2][9:]

            # logger.info('\n' + '=' * 50)
            # logger.info(query)

            # try:
            #     entitys_str = self.retrieval_judger.extract_entity(query)
            #     entitys = eval(entitys_str)
            #     reactants, products = entitys['reactants'], entitys['products']
            # except:
            #     reactants, products = self.retrieval_judger.extract_entity_by_re(query)
            reactants, products = self.retrieval_judger.extract_entity_by_re(query)
            # 判断尾节点在图谱中是否存在
            is_tail_node_exsit = True
            for product in products:
                tail_node = GraphDao(args).query_node('entity', name=product).first()
                if tail_node is None:
                    is_tail_node_exsit = False
                    break

            if is_tail_node_exsit:
                # 第一次检索(先开始逆向检索)
                self.args.use_reverse_retrieval = True
                pruned_fact_str, pruned_fact_triples, entropy_val, retrieved_fact_str, retrieved_fact_triple = self.rag_main(query, products, reactants)
                if str(entropy_val) != '[0.0]':
                    # 正向检索补救
                    self.args.use_reverse_retrieval = False
                    pruned_fact_str, pruned_fact_triples, entropy_val, retrieved_fact_str, retrieved_fact_triple = self.rag_main(query, reactants, products)

                    if str(entropy_val) != '[0.0]':
                        # 如果还是失败，添加历史信息，再检索一遍
                        self.args.use_history_info = True
                        pruned_fact_str, pruned_fact_triples, entropy_val, _, _ = self.rag_main(query, reactants, products)
                        self.args.use_history_info = False
                    if str(entropy_val) != '[0.0]':
                        # 补救失败，添加有用信息
                        pruned_fact_str, pruned_fact_triples, entropy_val = self.pruner.prune_fact(query, retrieved_fact_str, retrieved_fact_triple)
                        logger.info(f"entropy_val:{entropy_val}")
            else:
                # 尾节点不存在（只能一开始采取正向检索）
                pruned_fact_str, pruned_fact_triples, entropy_val, retrieved_fact_str, retrieved_fact_triple = self.rag_main(query, reactants, products)
                pruned_fact_str, pruned_fact_triples, entropy_val = self.pruner.prune_fact(query, retrieved_fact_str, retrieved_fact_triple)
                logger.info(f"entropy_val:{entropy_val}")

            if pruned_fact_str:
                pruned_fact_strs.append(pruned_fact_str)
                pruned_fact_triples_list.append(pruned_fact_triples)

            if self.args.need_gnn_model:
                graph, desc = self._get_subgraph(query, pruned_fact_triples_list)
                graph_list.append(graph)
                desc_list.append(desc)

        samples['desc'] = desc_list
        samples['graph'] = graph_list

        prune_score = 0
        try:
            if '[0.0]' == str(entropy_val):
                prune_score = 1
        except:
            logger.info(reactants)
        return samples, prune_score

    def rag_main(self, query, left_nodes, right_nodes):
        pruned_fact_str, pruned_fact_triples, entropy_val = '', '', []
        retrieved_fact_str, retrieved_fact_triple, is_prune_again = self.extracter.multi_hop_search(query, left_nodes, right_nodes)
        if retrieved_fact_triple:
            retrieved_fact_triple_split = retrieved_fact_triple.split('.\n')
            if len(retrieved_fact_triple_split) == 1 or not is_prune_again:
                pruned_fact_str = retrieved_fact_str
                pruned_fact_triples = retrieved_fact_triple_split
                entropy_val = [0.0]
                logger.info(f"entropy_val:{entropy_val}")
        return pruned_fact_str, pruned_fact_triples, entropy_val, retrieved_fact_str, retrieved_fact_triple

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def _get_subgraph(self, question, pruned_fact_triples_list):
        # 无需检索的，描述和图谱直接返回空值
        if not pruned_fact_triples_list:
            return Data(), ''

        # 问题编码
        q_emd = self.text2embedding([question])

        # 构建子图节点和边关系
        node_dict = {}
        edges = []
        for pruned_fact_triples in pruned_fact_triples_list:
            for pruned_fact_triple in pruned_fact_triples:
                head_node, edeg_attr, tail_node = pruned_fact_triple.split('|')

                if head_node not in node_dict:
                    node_dict[head_node] = len(node_dict)
                if tail_node not in node_dict:
                    node_dict[tail_node] = len(node_dict)
                edges.append({'head_node': node_dict[head_node], 'edge_attr': edeg_attr.strip(), 'tail_node': node_dict[tail_node], })
        nodes = pd.DataFrame(node_dict.items(), columns=['node_attr', 'node_id'])
        edges = pd.DataFrame(edges)

        desc = nodes.to_csv(index=False) + '\n' + edges.to_csv(index=False)

        # 节点与关系编码
        x = self.text2embedding(nodes.node_attr.tolist())
        e = self.text2embedding(edges.edge_attr.tolist())
        edge_index = torch.LongTensor([edges.head_node, edges.tail_node])

        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=e,
                    question_node=q_emd.repeat(x.size(0), 1),
                    question_edge=q_emd.repeat(e.size(0), 1),
                    num_nodes=len(nodes))

        return data, desc

    def text2embedding(self, texts):
        try:
            # Tokenize the list of texts
            encoding = self.embedding_tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

            # Move encoding to the appropriate device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            # Forward pass
            with torch.no_grad():
                embeddings = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
                # embeddings = outputs.last_hidden_state[:, 0, :]  # Assuming you want the CLS token embedding

            # Return the embeddings on CPU
            return embeddings.cpu()

        except Exception as e:
            logger.info(f"An error occurred: {e}")
            return torch.zeros((len(texts), 1024))


def main(args):
    # 步骤1：基本配置
    seed = args.seed

    seed_everything(seed=args.seed)
    logger.info(args)

    # 步骤2：数据集读取
    dataset = load_dataset[args.dataset]()

    train_dataset, val_dataset, test_dataset = dataset.get_train_eval_test_data_old()

    # 编写collate_fn函数
    # train_loader = DataLoader([train_dataset[i] for i in [54,55,408,409]], batch_size=args.process_batch, drop_last=True, pin_memory=True, shuffle=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=args.process_batch, drop_last=True, pin_memory=True, shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=args.process_batch, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)
    # test_loader = DataLoader([test_dataset[i] for i in [45, 64, 92, 96, 115]], batch_size=args.process_batch, drop_last=False, pin_memory=True, shuffle=False,
    #                          collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.process_batch, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # 步骤3：模型创建
    # args.llm_model_path = llama_model_path[args.llm_model_name]
    model = GraphLLMProcess(args=args)

    # 步骤4：训练
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = model.print_trainable_params()
    logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # 训练数据保存
    progress_bar = tqdm(range(len(train_loader)))
    train_data_save = []
    final_prune_score = 0
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        samples, prune_score = model(batch)
        final_prune_score += prune_score
        # 提取所有键和对应的值列表
        keys = samples.keys()
        values = zip(*samples.values())

        # 将每组值转换为字典
        result = [dict(zip(keys, group)) for group in values]

        train_data_save.extend(result)
        progress_bar.update(1)
    logger.info(f'训练剪枝正确率{final_prune_score / len(train_loader)}')
    torch.save(train_data_save, f'')
    logger.info('训练集保存完成')

    # 测试数据保存
    progress_bar = tqdm(range(len(test_loader)))
    test_data_save = []
    final_prune_score = 0
    for step, batch in enumerate(test_loader):
        optimizer.zero_grad()
        samples, prune_score = model(batch)
        final_prune_score += prune_score
        # 提取所有键和对应的值列表
        keys = samples.keys()
        values = zip(*samples.values())

        # 将每组值转换为字典
        result = [dict(zip(keys, group)) for group in values]

        test_data_save.extend(result)
        progress_bar.update(1)
    logger.info(f'测试剪枝正确率{final_prune_score / len(test_loader)}')
    torch.save(test_data_save, f'')
    logger.info('测试集保存完成')


def new_desc(desc):
    desc = desc[18:]
    cut_point = desc.find('head_node,edge_attr,tail_node')
    substrates = desc[:cut_point].split('\n')[:-2]
    relations = desc[cut_point + 29:].split('\n')[1:-1]

    required_substrates = []
    for substrate in substrates:
        required_substrates.append(substrate.split(',')[0])
    desc_correct = f'Please refer to the following information to answer,be careful not to reveal in your answer that you referenced the information I provided.\n' \
                   f'The required substrates are {" and ".join(required_substrates)}, involving a total of {len(relations)} reaction steps.\n'

    required_relations = []
    for relation in relations:
        required_relations.append(relation[3:-3])
    for idx, reaction_path in enumerate(required_relations):
        desc_correct += f'Step{idx + 1}:{reaction_path}\n'
    return desc_correct


def generate_120_test_data(file_path):
    """
    生成人工精修的120条测试数据
    直接按照规则生成，节省RAG检索的时间。
    todo：这写人工精修的内容，需要更新到图谱中的
    """
    model = GraphLLMProcess(args=args)
    prompt = """You are a professional organic chemist. Please provide a reaction step design based on the synthesis goal I have proposed. 

Question: [Specific issues.]

Please strictly follow the answer template below to generate answers。The answer template is as follows:
```
The required substrates are "XXX", "XXX", and "XXX", involving a total of [n] reaction steps.
Step 1: Using [substrate 1] and [substrate 2] as substrates, [xxx] as solvent, [xxx] as reagent, and [xxx] as catalyst, under [temperature, reaction time, pressure, pH, etc.] conditions, the final product [xxx] is generated, and the reaction equation is a chemical reaction equation in the form of [smiles].
Step 2: If this step is divided into multiple stages, it needs to be answered by stage. The first stage: [detailed reaction conditions], the second stage: [detailed reaction conditions], generates the final product [xxx], and the reaction equation is a chemical reaction equation in the form of [smiles].
……
Step n:
```

Be careful:
1. If there are no corresponding conditions, do not fabricate them. If there are other reaction conditions, they can also be supplemented.
2. Please output a paragraph for each step according to the natural language description, without dividing it into smaller parts.
            """

    def extract_reaction_equations(text):
        # 使用正则表达式匹配"the reaction equation is"后面的内容
        pattern = r"the reaction equation is\s+([^\n]+)"
        matches = re.findall(pattern, text)

        # 清理结果，去除可能的空格
        equations = []
        for match in matches:
            match = match.strip()
            if match[-1] == '.':
                equations.append(match[:-1].strip())
            else:
                equations.append(match)

        # 改进正则表达式：匹配Step开头到the reaction equation is之间的所有内容（包括多阶段）
        pattern = r"Step \d+.*?(?=\s*the reaction equation is)"

        matches = re.findall(pattern, text, re.DOTALL)

        # 清理结果，去除多余的空格和换行
        conditions = []
        for idx, match in enumerate(matches):
            cut_point = match.find(',')
            match = match[cut_point + 1:]
            match = match.strip()
            smile_condition = ' the reaction equation is ' + equations[idx].replace('△', '.')
            match += smile_condition
            conditions.append(match)

        assert len(equations) == len(conditions)
        return equations, conditions

    need_dicts = []
    with open(file_path, 'r', encoding='utf-8') as jsonfiles:
        datas = json.load(jsonfiles)
        for data in datas:
            question = data['question']
            answer = data['answer']
            steps = data['steps']
            id = data['testid']
            # if id == 108:
            #     print(1)
            print(id)
            smiles, conditions = extract_reaction_equations(answer)
            triples = []
            for idx, smile in enumerate(smiles):
                substances_str, products_str = smile.split('>>')
                substances = substances_str.split('△')
                products = products_str.split('△')
                triple = substances[0] + '|' + conditions[idx] + '|' + products[0]
                triples.append(triple)

            graph, desc = model._get_subgraph(question, [triples])
            desc = f'Please refer to the following information to answer,be careful not to reveal in your answer that you referenced the information I provided.\n{answer}'
            # desc = '\n' + desc + '\n' + 'The above is the knowledge graph content related to the question. Please carefully analyze the search content I provided and refer to it to provide your answer. Be careful to avoid using the following statements:\n-According to the provided information\n-Search content display\n-Mentioned in the knowledge base\n-Any citation format that may expose the source of information'

            need_dicts.append({"id": id,
                               "question": prompt.replace('[Specific issues.]', question),
                               "label": answer,
                               "desc": desc,
                               "graph": graph,
                               'steps': steps})

    torch.save(need_dicts, '')


if __name__ == '__main__':
    # loaded_samples = torch.load('train_data_30000.pt', weights_only=False)
    # with open('test_data_2000.json', 'w', encoding='utf-8') as f:
    #     for need_jsondict in loaded_samples:
    #         jsondict = {'id': need_jsondict['id'], 'label': need_jsondict['label'], 'question': need_jsondict['question']}
    #         json.dump(jsondict, f, ensure_ascii=False)
    #         f.write('\n')
    args = parse_args()

    # main(args)
    generate_120_test_data('')

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
