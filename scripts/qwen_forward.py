"""Qwen2.5-0.5B prefill + two decode steps through torch.compile(inductor, cpu_backend="triton").

Model script for scripts/tune_gemm_on_board.py (any script that runs a compiled model works).
Env: QWEN_PROMPT, QWEN_MAX_CACHE_LEN.
"""
import os

import torch
import torch._inductor.config as inductor_config
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

inductor_config.cpu_backend = "triton"
inductor_config.max_autotune = False
inductor_config.max_autotune_gemm = False

MAX_CACHE_LEN = int(os.environ.get("QWEN_MAX_CACHE_LEN", "128"))
PROMPT = os.environ.get("QWEN_PROMPT", "Please briefly explain what Time to First Token (TTFT) is.")

name = os.path.join(os.environ["HOME"], "qwen2.5-0.5b-instruct")
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="eager").eval()
cache = StaticCache(config=model.config, max_cache_len=MAX_CACHE_LEN)
model.forward = torch.compile(model.forward, backend="inductor", dynamic=False)

ids = tok(PROMPT, return_tensors="pt").input_ids
L = ids.shape[1]
with torch.no_grad():
    out = model(input_ids=ids, past_key_values=cache, cache_position=torch.arange(L), use_cache=True)
    for step in range(2):
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        out = model(input_ids=nxt, past_key_values=cache, cache_position=torch.tensor([L + step]), use_cache=True)
print("model script finished")
