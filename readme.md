# ChemKGL: Bridging Knowledge Graphs and Large Language Models for Chemical Multi-Step Reaction Pathway Inference

## Abstract

Large language models (LLMs) have shown promising potential in chemistry, with prior work exploring molecular recognition, classification, and property prediction. Despite the achieved progress, LLMs are still far from satisfactory when dealing with complex chemical multi-step reaction pathway inference task due to the lack of domain knowledge and limited ability to maintain consistent multi-step reasoning. To address these challenges, we propose ChemKGL, a novel multi-step reasoning framework enhanced by knowledge graph retrieval. We reformulate the task as a knowledge graph retrieval problem to better control LLM generation, and leverage graph technologies to integrate textual descriptions, reaction conditions, and materials. Based on this reformulation, we design a novel multi-retrieval strategy to improve the performance of reaction pathway inference, which includes forward retrieval, reverse retrieval, and historical retrieval. An entropy-based pruning mechanism is further introduced to alleviate cold-start issues caused by retrieval failure. In addition, we construct a comprehensive chemical knowledge graph and corresponding dataset to support model training and analysis. Finally, we employ multiple evaluation methods to assess our proposed ChemKGL, and the experimental results demonstrate its superiority. Our code and dataset can be obtained at https://anonymous.4open.science/r/ChemKGL.

------

## 📦 数据说明

### ✅ 已开源：

- 专家标注的 120 条多步化学反应测试集（测试评估用）
  - test_data_120.pt
  - test_data_120_EBP.pt
  - test_data_120_one_shot.pt
- 训练集前20条
  - train_data_20.pt
- 少量用于构建知识图谱的三元组数据
  - KG_triple.csv

### ❌ 未开源（可申请或自定义）：

- 完整训练集
- 2000 条大规模测试集
- 完整的知识图谱数据

> 你需要按如下需求构建自己的数据集和知识图谱，开源的代码中只提供了少量的基本数据，供代码调试

------

## 📘 知识图谱构建

### 1. 构建三元组格式的反应信息

每条记录需满足如下格式（建议以 `.csv` 保存，每行一条）：

```
反应物|products|产物|具体的反应描述
```

#### 示例：

```
c1cn[nH]c1|products|Brc1cccc(-n2cccn2)n1|reacts with Brc1cccc(Br)n1 to generate Brc1cccc(-n2cccn2)n1,with the reaction conditions being: 1,4-dioxane as solvent, potassium tert-butylate as reagent, temperature Reflux, reaction time 48 h.
```

具体可见`KG_triple.csv`文件。

### 2. 注入 Neo4j 数据库

运行以下命令将上述三元组导入 Neo4j：

```bash
python dataset/construct_KG.py
```

- **Neo4j版本要求**：`neo4j-community-5.26.18-unix`

------

## 📚 训练集与测试集生成

### 1. 多步反应 Q&A 生成

每个多步反应案例生成如下四种问题形式（以 `products` 和 `primary_substrates` 为变量）：

```python
question1 = f'I would like to obtain the reaction product {str(products)}. Given the required partial reaction substrates {str(primary_substrates)},please help me supplement the remaining required reaction substrates and provide the reaction pathway as well as the conditions required for the reaction.'

question2 = f'To synthesize compound {str(products)}, substrates {str(primary_substrates)} have been provided as initial materials.Please determine the additional substrates needed to complete the reaction sequence, along with the detailed mechanistic pathway and specific experimental conditions required for each step.'

question3 = f'Given substrates {str(primary_substrates)} as starting materials, propose the missing reagents required for the synthesis of product {str(products)}.Include a complete reaction mechanism and provide reaction conditions for each stage.'

question4 = f'The goal is to produce the target substance {str(products)} using substrates {str(primary_substrates)}. Determine the required supplementary reagents for the reaction sequence and provide a step-by-step approach with relevant conditions to achieve optimal synthesis.'

questions = [question1, question2, question3, question4]
```

### 2. 答案模板格式：

```python
answer = f'The required substrates are {" and ".join(required_substrates)}, involving a total of {len(multisteps)} reaction steps.\n
Step 1：……\n
Step 2：……\n
……
Step n：……'
```

### 3. 数据结构要求：

每条 Q&A 的数据至少需要包含以下字段：

```python
data_dict = {'reactionid': reaction_id,
             'question': question,
             'answer': answer,
             'steps': steps}
```

具体的问题和答案的数据形式，可以参考`train_data_20.pt`和`test_data_120.pt`

### 4. 子图生成与融合：

通过以下命令检索知识图谱子图信息并嵌入：

```bash
python dataset/data_generate_include_text_and_gnn.py
```

输出将包含：

- 基于检索的子图文本
- GAT 网络提取的结构化图信息

------

## 🧠 模型训练与推理

### 1. 模型训练

```bash
bash train.sh
```

你可以在.sh文件中调整参数，也可以在utils/config.py文件中修改参数。

将训练并保存模型，用于后续推理。

### 2. 推理命令

- **本地可部署对比模型推理**

```bash
python inference_llama8.py     # LLaMA3.1-8B
python inference_qwen7.py      # Qwen2.5-7B
python inference_qwen4.py      # LLaMA3.1-4B
python inference_qwen8.py      # LLaMA3.1-8B
```

- **Deepseek-v3、Qwen2.5-72B：**
   - 本研究是购买相应模型的推理服务，并调用接口运行推理。
   - 算力允许可以本地部署

------

## 🔬 消融实验设置

### 1. **LLaMA3.1-no-gnn**

- 不使用结构化子图信息，仅使用`test_data_120.pt`中的`desc`字段，仅将文本子图作为 RAG 材料输入 LLaMA3.1-8B

### 2. **LLaMA3.1-EBP**

- 令`args.tail_node_not_exist=False`

- 使用test_data_120_EBP.pt数据进行推理。

### 3. **LLaMA3.1-one-shot**

- 使用test_data_120_one_shot.pt数据进行推理了。

Qwen系列同理。

### 4.混合检索准确率

调整config文件参数配置

- 集束搜索：`beam_width = 1 or beam_width=3`。
- 混合检索方式：`retrieval_style:0:混合检索。1：逆向检索。2.正向检索。3.历史检索`

------

## 📌oMeBench与rekgmcts

每个文件夹中有单独的readme.md文件，具体代码运行可参考。

oMeBench需要单独构建图谱，运行`\oMeBench-main\scripts\generate_graph.py`即可

---------------

## 📊 评估指标与评分

我们使用 GPT4-O1、Gemini 2.5 Pro、Deepseek-R1 接口进行自动化评估，你需要自己购买接口服务。

- 请参考论文附录提供的评分提示模板（Prompt）进行自动评估。
- 计算不同模型在多步化学反应推理任务中的评分。

---------------

## 📣 附录
IJCAI__ECAI_26_ChemKGL__Camera_Ready__appendix.pdf为原始论文的Appendix文件，若需要阅读论文的附录部分，可以参考此文件。
