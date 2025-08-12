# from model.llm import LLM
from model.graph_llm import GraphLLM


load_model = {
    # 'llm': LLM,
    # 'inference_llm': LLM,
    'graph_llm': GraphLLM,
}

# Replace the following with the model paths
llama_model_path = {
    '7b': 'meta-llama/Llama-2-7b-hf',
    '13b': 'meta-llama/Llama-2-13b-hf',
    '7b_chat': 'meta-llama/Llama-2-7b-chat-hf',
    '13b_chat': 'meta-llama/Llama-2-13b-chat-hf',
    'lama-3.1-8b': '/cogbrain/zmliu7/Meta-Llama-3.1-8B-Instruct/',
    'qwen': '/cogbrain/zmliu7/Qwen2___5-7B-Instruct/',
}
