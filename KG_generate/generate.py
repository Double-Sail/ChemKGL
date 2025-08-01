import json
import os
import re


def generate_main(file_path):
    with open(file_path, 'r', encoding='utf-8') as files:
        for file in files:
            json_line = json.loads(file)
            answer = json_line['answer']

            cut_point = answer.find("步骤 6")
            if cut_point != -1:
                answer = answer[cut_point + 9:]
                # print(answer)
            else:
                print("未找到 '步骤 6' 相关内容")

            json_pattern = r'```json\n(.*?)```'
            json_match = re.search(json_pattern, answer, re.DOTALL)

            # 提取 JSON 字符串
            if json_match:
                json_text = json_match.group(1)
                # 解析 JSON 字符串
                data = json.loads(json_text)
                print(data)
            else:
                print("未找到 JSON 部分")
            print(1)


if __name__ == '__main__':
    file_path = 'HEA_cat1_n10_CJAD-T4PL-10C0-SPE1.json'
    generate_main(file_path)
