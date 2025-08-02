import argparse
import random, os
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="MyRAG")

    parser.add_argument("--model_name", type=str, default='graph_llm')
    parser.add_argument("--project", type=str, default="projection")
    parser.add_argument("--seed", type=int, default=3)

    parser.add_argument("--dataset", type=str, default='mydata')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=5)

    parser.add_argument("--device", type=int, default=4)

    # Model Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_steps", type=int, default=2)

    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=float, default=1)

    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=8)

    # LLM related
    # parser.add_argument("--llm_model_path", type=str, default='/cogbrain_data/fanyang58/models/Meta-Llama-3.1-8B-Instruct/')
    # parser.add_argument("--llm_model_name", type=str, default='lama-3.1-8b')
    parser.add_argument("--llm_model_path", type=str, default='Qwen2___5-7B-Instruct/')
    parser.add_argument("--llm_model_name", type=str, default='Qwen7')
    # parser.add_argument("--embedding_model_name", type=str, default='sbert')
    parser.add_argument("--embedding_model_path", type=str, default='/all-roberta-large-v1/sentence-transformers/all-roberta-large-v1')
    parser.add_argument("--llm_frozen", type=str, default='True')
    parser.add_argument("--llm_num_virtual_tokens", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default='/code/KGLLM2.0/MyRAG/output')
    parser.add_argument("--max_txt_len", type=int, default=1600)
    parser.add_argument("--max_new_tokens", type=int, default=1600)

    # GNN related
    parser.add_argument("--need_gnn_model", type=bool, default=True)
    parser.add_argument("--gnn_model_name", type=str, default='gat')
    parser.add_argument("--gnn_num_layers", type=int, default=4)
    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--alignment_mlp_layers", type=int, default=3)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--distance_operator", type=str, default='euclidean')
    parser.add_argument("--gnn_dropout", type=float, default=0.0)

    # neo4j related
    parser.add_argument("--neo4j_url", type=str)
    parser.add_argument("--neo4j_username", type=str)
    parser.add_argument("--neo4j_password", type=str)

    # prune
    parser.add_argument('--loss', type=str, default='prob_div_log')
    parser.add_argument('--beam_width', type=int, default=3)
    parser.add_argument('--jump_num', type=int, default=5)
    parser.add_argument('--process_batch', type=int, default=1)

    parser.add_argument('--use_history_info', type=bool, default=False)
    parser.add_argument('--use_reverse_retrieval', type=bool, default=True)

    args = parser.parse_args()
    return args


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
