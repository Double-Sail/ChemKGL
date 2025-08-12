from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import json

# 配置参数
MODEL_PATH = "Qwen3-8B"
DATA_PATH = "dataset/mydataset/test_data_2000.pt"
BATCH_SIZE = 8  # 根据显存调整
MAX_NEW_TOKENS = 8192
DEVICE_MAP = "cuda:1"

# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE_MAP
)

# 加载并预处理数据
test_datas = torch.load(DATA_PATH, weights_only=False)
valid_datas = [data for data in test_datas if data.get("desc", "") != ""]


def batch_inference(batch_data):
    """批量推理核心函数"""
    # 提取有效字段
    prompts = [data["question"] for data in batch_data]
    ids = [data["id"] for data in batch_data]

    # 生成对话模板
    batch_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        ) for prompt in prompts
    ]

    # 批量编码
    model_inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding="longest",  # 动态填充
        truncation=True,
        max_length=4096
    ).to(model.device)

    # 批量生成
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True  # 启用采样生成
    )

    # 解析结果
    results = []
    for i, (input_seq, output_seq) in enumerate(zip(model_inputs.input_ids, generated_ids)):
        # 截取生成部分
        output_ids = output_seq[len(input_seq):].tolist()

        # 解析thinking标记
        try:
            split_pos = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            split_pos = 0

        # 解码内容
        thinking = tokenizer.decode(output_ids[:split_pos], skip_special_tokens=True).strip()
        answer = tokenizer.decode(output_ids[split_pos:], skip_special_tokens=True).strip()

        results.append({
            "id": ids[i],
            "question": prompts[i],
            "thinking": thinking,
            "answer": answer
        })

    return results


# 执行批量推理
need_dicts = []
for i in tqdm(range(0, len(valid_datas), BATCH_SIZE), desc="Processing Batches"):
    batch = valid_datas[i:i + BATCH_SIZE]
    need_dicts.extend(batch_inference(batch))

# 保存结果
with open("output/mydata/qwen8_no_rag_2048.json", "w", encoding="utf-8") as f:
    for item in need_dicts:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')
