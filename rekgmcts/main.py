# src/main.py
import argparse
import yaml
import json
from runner import Runner
import torch
import re
import time
import os
from tqdm import tqdm

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def extract_entity_by_re(query):
    # 提取目标物质
    matches = re.findall(r"\['(.*?)']", query)
    if query.strip().startswith('Given substrates'):
        substrates = matches[0]
        products = matches[1]
    else:
        substrates = matches[1]
        products = matches[0]
    substrates_str = substrates.replace("'", "").replace(" ", "").replace("\\\\", "\\")
    products_str = products.replace("'", "").replace(" ", "").replace("\\\\", "\\")
    return substrates_str.split(','), products_str.split(',')
    
def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/MyRAG/rekgmcts/default.yaml")
    parser.add_argument("--question", type=str, default="Who influenced Albert Einstein?")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--demo", action="store_true", help="Run in demo (in-memory) mode without Neo4j/OpenAI")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    runner = Runner(cfg, demo_mode=args.demo)

    datas = torch.load('/MyRAG/dataset/mydataset/test_data_120.pt', weights_only=False)

    count = 0
    errors = []
    for data in tqdm(datas, total=len(datas), desc='🚀 mcts子图检索', unit='entry'):
        question = data['question']
        id = data["id"]
        steps = data['steps']
        runner.cfg['mcts']['max_depth'] = steps

        if id < 89:
            continue
        try:
            substrates, products = extract_entity_by_re(question)
            result = runner.run_question(question, substrates, seed_entity_id=args.seed)
        except Exception as e:
            text = f'检索失败，数据id:{id}，错误原因: {e}'
            print(text)
            errors.append(text)
            continue
        
        data['desc'] = result

        # 计算检索准确率
        if products[0] in result:
            count += 1
            print(f'已完成{count}条数据。')


    end_time = time.time()
    total_cost = end_time - start_time  # 总耗时(秒)

    acc_info = f'count={count},检索准确率{count/len(datas):.4f}'
    time_info = f'✅ 程序整体运行完成，总耗时: {total_cost:.2f} 秒'
    print(acc_info)
    print(time_info)
    torch.save(datas, f'/MyRAG/dataset/mydataset/test_data_{len(datas)}_mcts.pt')

    # ========== 核心修改：整合错误信息到日志 ==========
    log_file_path = "/MyRAG/rekgmcts/rag_run_log.txt"
    # 拼接日志内容：时间戳 + 准确率 + 耗时 + 错误统计 + 所有错误详情
    log_head = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {acc_info}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {time_info}\n"
    error_count_info = f"本次运行共出现 {len(errors)} 条检索失败记录：\n"
    error_detail = '\n'.join(errors) if errors else "无检索失败记录\n"
    log_content = log_head + error_count_info + error_detail + f"{'='*80}\n\n"
    
    with open(log_file_path, 'a+', encoding='utf-8') as f:
        f.write(log_content)


if __name__ == "__main__":
    main()