import transformers
import torch
import json
from tqdm import tqdm

# 初始化模型
model_id = "Meta-Llama-3.1-8B-Instruct/"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'left'
# 在tokenizer初始化之后添加
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 使用eos_token作为pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 优化后的模型加载配置
model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "device_map": "cuda:3"
}

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    model_kwargs=model_kwargs,
)


def batch_inference(batch_data):
    """批处理推理函数"""
    # 提取问题和ID
    prompts = [data["question"] for data in batch_data]
    # prompts = [data["question"] + data['desc'] for data in batch_data]
    ids = [data["id"] for data in batch_data]

    messages = [
        # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        [{"role": "user", "content": prompt}] for prompt in prompts
    ]

    # 批量生成
    outputs = pipeline(
        messages,
        max_new_tokens=4096,
        batch_size=len(batch_data),
        do_sample=True
    )

    # 组装结果
    return [
        {
            "answer": output[0]["generated_text"][-1]['content'],
            "id": ids[i],
            "model_name": "llama8_only",
            "question": batch_data[i]["question"]
        } for i, output in enumerate(outputs)
    ]


# 主处理流程
test_datas = torch.load('dataset/mydataset/test_data_120_one_shot.pt',
                        weights_only=False)

# 过滤无效数据（提前过滤提升效率）
valid_datas = [data for data in test_datas if data["desc"] != ""]

# 动态批处理
batch_size = 8
need_dicts = []

with torch.inference_mode():  # 显存优化
    for i in tqdm(range(0, len(valid_datas), batch_size),
                  desc=f'Batch Processing (size={batch_size})'):
        batch = valid_datas[i:i + batch_size]
        need_dicts.extend(batch_inference(batch))

# 结果保存
with open('output/mydata/llama8_one_shot.json',
          'w', encoding='utf-8') as f:
    for item in need_dicts:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')
