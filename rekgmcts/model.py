# src/model.py
import json
import random
import re
from typing import Any, List, Dict

import prompts, utils

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
import torch


# ======================================================
# Base class
# ======================================================

class BaseLLM:
    def score_policy(self, prompt: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def score_value(self, prompt: str) -> float:
        raise NotImplementedError


# ======================================================
# DummyLLM (baseline / debug)
# ======================================================

class DummyLLM(BaseLLM):
    """
    用途：
    1) 在没有真实 LLM 时跑通整个 MCTS + KG 管线
    2) 作为 baseline，对比 LLM-guided 搜索是否真的有提升
    """

    def score_policy(self, prompt: str):
        lines = [l for l in prompt.splitlines() if l.strip() and l.strip()[0].isdigit()]
        if not lines:
            return []

        uniform = 1.0 / len(lines)
        results = []

        for ln in lines:
            try:
                nid = int(ln.split("nid=")[-1].split(")")[0])
            except Exception:
                nid = random.randint(1000, 9999)

            rel = "unknown"
            name = "unknown"

            if "rel=" in ln:
                rel = ln.split("rel=")[1].split(",")[0].strip()
            if "ent=" in ln:
                name = ln.split("ent=")[1].split("(")[0].strip()

            results.append({
                "rel": rel,
                "name": name,
                "nid": nid,
                "score": uniform
            })

        return results

    def score_value(self, prompt: str) -> float:
        m = re.search(r"nodes=\[?([0-9, ]+)\]?", prompt)
        if m:
            n_nodes = len([x for x in m.group(1).split(",") if x.strip().isdigit()])
        else:
            n_nodes = 1

        v = min(0.9, 0.1 + 0.15 * n_nodes)
        return max(0.0, min(1.0, v + random.uniform(-0.05, 0.05)))


# ======================================================
# Local HuggingFace LLM (LLaMA / Qwen)
# ======================================================

class LocalHFLLM(BaseLLM):
    """
    支持本地：
    - LLaMA 3 / 2
    - Qwen2 / Qwen2.5 / Qwen3
    """

    def __init__(
        self,
        model_path: str,
        device: str,
        max_new_tokens: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map=device if "cuda" in device else None
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    def _generate(self, prompt: str) -> str:
        out = self.pipe(prompt)[0]["generated_text"]
        return out[len(prompt):].strip()

    # --------------------------------------------------
    # policy head
    # --------------------------------------------------
    def score_policy(self, prompt: str) -> List[Dict[str, Any]]:
        raw = self._generate(prompt)
        parsed = utils.parse_json_safe(raw)

        if parsed is None or not isinstance(parsed, list):
            return []  # fallback handled by caller
        return parsed

    # --------------------------------------------------
    # value head
    # --------------------------------------------------
    def score_value(self, prompt: str) -> float:
        raw = self._generate(prompt)
        try:
            return float(raw.strip())
        except Exception:
            m = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", raw)
            if m:
                return float(m.group(0))
            return 0.0
