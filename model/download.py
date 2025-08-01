# 模型下载
from modelscope import snapshot_download

# model_dir = snapshot_download('sentence-transformers/all-roberta-large-v1', cache_dir='D:\Documents\model')

model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='D:\Documents\model')
