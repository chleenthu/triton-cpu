"""x86 port of the GPU box's scripts/qwen_forward.py (chlee@140.114.78.168:~/triton/scripts):
one Qwen2.5-0.5B forward pass, no KV cache, HF eager attention, torch.compile(inductor)
with cpu_backend="triton". Same input as the GPU run: torch.manual_seed(0), 128 random
token ids, all-ones attention mask. bf16 instead of the GPU's fp16.
"""
import argparse
import os

import torch
import torch._inductor.config as inductor_config
from transformers import AutoModelForCausalLM

inductor_config.cpu_backend = "triton"

MODEL = os.path.join(os.environ["HOME"], "qwen2.5-0.5b-instruct")


class ForwardOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False, return_dict=False)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager").eval()
    compiled = torch.compile(ForwardOnly(model).eval(), backend="inductor",
                             fullgraph=False, dynamic=False)

    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (1, args.seq_len), dtype=torch.long)
    attention_mask = torch.ones((1, args.seq_len), dtype=torch.long)
    with torch.inference_mode():
        logits = compiled(input_ids, attention_mask)
    print("Logits shape:", tuple(logits.shape))
    print("Done")


if __name__ == "__main__":
    main()
