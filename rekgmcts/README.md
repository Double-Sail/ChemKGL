# ReKG-MCTS: Monte Carlo Tree Search for Knowledge Graph Reasoning with LLMs

Codebase for our ACL submission "ReKG-MCTS: Integrating Monte Carlo Tree Search with LLMs for Knowledge Graph Reasoning"

## 📖 Repository Overview

This repository implements a knowledge graph reasoning framework based on Monte Carlo Tree Search (MCTS) and Large Language Models (LLMs). The core idea is to:
1. **Use MCTS for structured search**: Explore paths on the knowledge graph.
2. **Leverage LLMs for semantic guidance**: Act as a policy model (selecting expansion directions) and a value evaluator (scoring path quality).
3. **Iterative optimization in four phases**: Node Selection → Path Expansion → Simulation → Value Backpropagation.

## 📂 Code Structure

### Core Modules

| File | Description |
|------|-------------|
| `mcts.py` | Core MCTS implementation<br>✅ Node definition (MCTSNode)<br>✅ Search process (MCTSPathFinder)<br>✅ UCB selection / Path expansion / Simulation backpropagation |
| `model.py` | LLM interface encapsulation<br>✅ HuggingFace model loading (HUGGINGFACE_LLM)<br>✅ GPT-4 API interface (GPT4_LLM)<br>✅ Answer generation and verification |
| `sparql_utils.py` | Knowledge graph interaction<br>✅ SPARQL query execution<br>✅ Entity and relation retrieval<br>✅ Freebase format processing |
| `utils.py` | Utility functions<br>✅ Answer extraction<br>✅ Relation cleaning<br>✅ Entity name resolution |
| `prompt.py` | Prompt template management<br>✅ Relation extraction prompts<br>✅ Path evaluation prompts<br>✅ Answer generation prompts |

## 🚀 Quick Start

### Environment Setup

1. **Install and Deploy Freebase Database**
**Requirements:**
- OpenLink Virtuoso 7.2.5 ([download](https://sourceforge.net/projects/virtuoso/files/virtuoso/))
- Python 3
- Freebase dump ([download](https://developers.google.com/freebase?hl=en))

**Setup Steps:**
```shell
# 1. Data Preprocessing
gunzip -c freebase-rdf-latest.gz > freebase  # Decompress dump (400G)
nohup python -u FilterEnglishTriplets.py 0<freebase 1>FilterFreebase 2>log_err &  # Clean data (output: 125G)

# 2. Virtuoso Installation
tar xvpfz virtuoso-opensource.x86_64-generic_glibc25-linux-gnu.tar.gz
cd virtuoso-opensource/database/
mv virtuoso.ini.sample virtuoso.ini

# 3. Start Virtuoso Service
../bin/virtuoso-t &  # Run in background
../bin/isql 1111 dba dba  # Connect to database

# 4. Import Data (Execute in isql console)
SQL> ld_dir('.', 'FilterFreebase', 'http://freebase.com');
SQL> rdf_loader_run();
```

2. **Configure SPARQL Endpoint**
```python
# Modify in sparql_utils.py
SPARQLPATH = "http://localhost:8890/sparql"  # Default Virtuoso endpoint
```

3. **Verify Database Connection**
```python
# Run the test script from Freebase Setup section
python test_sparql.py
```

4. **Configure model paths (e.g., Llama3)**
```python
# Configure model paths in main.py
MODEL_PATHS = {
    "Llama3-8B": "/path/to/Llama-3.8B",
}
```

### Running the Example

Basic execution command:
```bash
python main.py \
    --model Llama3-8B \
    --data cwq \
    --exploration_constant 0.5 \
    --depth 3 \
    --width 5 \
    --iterations 3 \
```

Key parameters:
- `--model`: Supported LLM types (Llama3-8B/70B, Qwen-7B, etc.)
- `--data`: Dataset selection (cwq/webqsp)
- `--exploration_constant`: UCB exploration coefficient
- `--depth`: Maximum mcts depth
- `--width`: Maximum mcts width
- `--iterations`: Maximum mcts iterations


安装与运行说明

在虚拟环境中安装依赖：

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt


配置（如果你要连接真实 Neo4j）：

编辑 configs/default.yaml 中的 neo4j.uri/user/password。

如果使用 OpenAI，设置环境变量 OPENAI_API_KEY 并把 model.type 设为 "openai"。

运行 Demo（推荐先 demo 模式快速验证）：

python -m src.main --question "Who influenced Albert Einstein?" --demo


示例输出（demo 模式，可能略有差异）：

{
  "nodes": [1, 2, 3],
  "edges": [
    [1, "influenced_by", 2],
    [1, "affiliated_with", 3]
  ]
}


连接真实 Neo4j（真实实验）：

# 确保 Neo4j 服务运行并且配置正确
python -m src.main --question "Who influenced Albert Einstein?" 
# (如果你的 config 指向 Neo4j, 默认不是 demo mode)


使用 OpenAI model：

在 configs/default.yaml 把 model.type 改为 openai 并设置 OPENAI_API_KEY 环境变量:

export OPENAI_API_KEY="sk-..."
python -m src.main --question "Who influenced Albert Einstein?"

关于可扩展性与改进建议（实务要点）

缓存 Neo4j 查询：在大量 MCTS 模拟期间频繁访问同一节点会产生大量查询。建议给 get_neighbors 加本地缓存（LRU）。

并行化模拟：当前实现是同步的。可以把多个 MCTS rollouts 并行化（注意 OpenAI rate-limit 或 Neo4j 并发限制）。

更强的 policy/value 合并：把 LLM 的 prior (policy score) 与历史 visit counts 合并为 PUCT 的 prior；对 LLM 返回不稳定的情况做温和平滑。

更严格的输出格式：要求 LLM 返回严格 JSON（在 prompt 中强调“Only return JSON”），并用 utils.parse_json_safe 解析。

归一化评分：若你使用 SPARQL/Cypher 匹配结果作为得分来源，与 LLM 得分混合时请先归一化到 [0,1]。

可视化：用 networkx 或 pyvis 将最终子图可视化，便于人工检查。