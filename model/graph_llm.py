import contextlib
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from model.gnn import load_gnn_model
from prune.prune import Extract, Prune
from torch_geometric.data.data import Data
import pandas as pd
import logging
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import os

logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(f'{os.getcwd()}/logs/train.log', mode='w')])
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


class Sentence_Transformer(nn.Module):

    def __init__(self, pretrained_repo):
        super(Sentence_Transformer, self).__init__()
        print(f"inherit model weights from {pretrained_repo}")
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


class GraphLLM(torch.nn.Module):

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
        if self.tokenizer.pad_token is None:
            # 按照这种形式设置pad，不然会输出很多额外信息（结束符号无限循环）
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.tokenizer.pad_token_id = 0

        # 默认使用llama模型，dtype给的是float16
        if args.llm_model_name.startswith('Qwen'):
            self.tokenizer.padding_side = 'right'
            dtype = torch.bfloat16
        else:
            self.tokenizer.padding_side = 'left'
            dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing Model")
            for _, param in model.named_parameters():
                param.requires_grad = False

        self.model = model
        print(f'Finish loading {self.args.llm_model_path}!')

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
        if args.llm_model_name.startswith('Qwen'):
            if '4B' in args.llm_model_name:
                self.projector = nn.Sequential(
                    nn.Linear(args.gnn_hidden_dim, 2048),
                    nn.GELU(),
                    nn.Linear(2048, 2560),
                ).to(self.model.device)
            elif '8B' in args.llm_model_name:
                self.projector = nn.Sequential(
                    nn.Linear(args.gnn_hidden_dim, 2048),
                    nn.GELU(),
                    nn.Linear(2048, 4096),
                ).to(self.model.device)
            else:
                # qwen2.5 7b
                self.projector = nn.Sequential(
                    nn.Linear(args.gnn_hidden_dim, 2048),
                    nn.GELU(),
                    nn.Linear(2048, 3584),
                ).to(self.model.device)
        else:
            # 默认使用llama3.1 8b
            self.projector = nn.Sequential(
                nn.Linear(args.gnn_hidden_dim, 2048),
                nn.GELU(),
                nn.Linear(2048, 4096),
            ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

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
        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False)
        if 'lama' in self.args.llm_model_name:
            bos_embeds = self.word_embedding(
                self.tokenizer(self.tokenizer.bos_token, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples['graph'])
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            if len(graph_embeds) != batch_size and i >= batch_size - 1:
                break
            # Add bos & eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = questions.input_ids[i] + descriptions.input_ids[i][
                                                 :self.max_txt_len] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            if 'lama' in self.args.llm_model_name:
                inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            else:
                inputs_embeds = torch.cat([graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            if len(graph_embeds) != batch_size and i >= batch_size - 1:
                break
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        # with self.maybe_autocast():
        with torch.autocast(device_type='cuda', dtype=self.model.dtype):
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        return outputs.loss

    def inference(self, samples):
        # encode description, questions and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(self.tokenizer.eos_token, add_special_tokens=False)
        if 'lama' in self.args.llm_model_name:
            bos_embeds = self.word_embedding(
                self.tokenizer(self.tokenizer.bos_token, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples["graph"])
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # Add bos & eos token
            input_ids = questions.input_ids[i] + descriptions.input_ids[i][
                                                 :self.max_txt_len] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            if 'lama' in self.args.llm_model_name:
                inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            else:
                # 千问系列，没有bos，是right padding
                inputs_embeds = torch.cat([graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

            # inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        # 推理时max_new_tokens尽量给大一点
        with torch.autocast(device_type='cuda', dtype=self.model.dtype):
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=2048,
                eos_token_id=self.tokenizer.eos_token_id,
                attention_mask=attention_mask,
                do_sample=True,
                # early_stopping=True,
                use_cache=True  # IMPORTANT!
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc']}

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
                edges.append({'head_node': node_dict[head_node], 'edge_attr': edeg_attr.lower().strip(),
                              'tail_node': node_dict[tail_node], })
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
            print(f"An error occurred: {e}")
            return torch.zeros((len(texts), 1024))
