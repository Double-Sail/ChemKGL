from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm

# 配置参数
MODEL_PATH = "Qwen2___5-7B-Instruct"
DATA_PATH = "dataset/mydataset/test_data_120_one_shot.pt"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 4096
DEVICE_MAP = "cuda:3"

# 初始化模型组件
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE_MAP
)

# 确保填充符配置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


def batch_inference(batch_data):
    """核心批量推理逻辑"""
    # 构造批量模板
    batch_texts = []
    for data in batch_data:
        messages = [
            # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": data["question"]}
            # {"role": "user", "content": data["question"] + data["desc"]}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        batch_texts.append(text)

    # 编码与填充
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
        do_sample=True
    )

    # 解析结果
    batch_results = []
    for i, (input_ids, output_ids) in enumerate(zip(model_inputs.input_ids, generated_ids)):
        generated_part = output_ids[len(input_ids):].tolist()
        response = tokenizer.decode(generated_part, skip_special_tokens=True).strip()

        batch_results.append({
            "id": batch_data[i]["id"],
            "question": batch_data[i]["question"],
            "answer": response
        })

    return batch_results


# 主处理流程
if __name__ == "__main__":
    # 加载并过滤数据
    test_datas = torch.load(DATA_PATH, weights_only=False)
    valid_datas = [data for data in test_datas if data.get("desc", "") != ""]

    # 分批次处理
    need_dicts = []
    for i in tqdm(range(0, len(valid_datas), BATCH_SIZE), desc="Processing"):
        batch = valid_datas[i:i + BATCH_SIZE]
        try:
            results = batch_inference(batch)
            need_dicts.extend(results)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"OOM at batch {i // BATCH_SIZE}, reducing batch_size to {max(1, BATCH_SIZE // 2)}")
                BATCH_SIZE = max(1, BATCH_SIZE // 2)
                continue
            else:
                raise e

    # 保存结果
    with open("output/mydata/qwen7_one_shot.json", "w", encoding="utf-8") as f:
        for item in need_dicts:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
