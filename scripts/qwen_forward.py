import argparse

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class ForwardOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )
        return outputs[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seq-len",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--attention",
        type=str,
        default="eager",
        choices=[
            "eager",
            "hf_flash2",
            "flash_attention_2",
        ],
    )

    args = parser.parse_args()

    seq_len = args.seq_len

    print("PyTorch   :", torch.__version__)
    print("GPU       :", torch.cuda.get_device_name(0))
    print("CC        :", torch.cuda.get_device_capability(0))
    print("Model     :", MODEL_ID)
    print("Seq len   :", seq_len)
    print("Attention :", args.attention)

    # ------------------------------------------------------------
    # Select Hugging Face attention interface
    # ------------------------------------------------------------

    if args.attention == "hf_flash2":
        hf_attention = "kernels-community/flash-attn2"

    elif args.attention == "flash_attention_2":
        hf_attention = "flash_attention_2"

    else:
        hf_attention = "eager"

    print("HF attn   :", hf_attention)

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation=hf_attention,
    )

    model.eval()
    model.to("cuda")

    wrapper = ForwardOnly(model).eval()

    # ------------------------------------------------------------
    # torch.compile -> TorchDynamo -> TorchInductor
    # ------------------------------------------------------------

    compiled = torch.compile(
        wrapper,
        backend="inductor",
        fullgraph=False,
        dynamic=False,
    )

    # ------------------------------------------------------------
    # Fixed deterministic input
    # ------------------------------------------------------------

    torch.manual_seed(0)

    input_ids = torch.randint(
        low=0,
        high=model.config.vocab_size,
        size=(1, seq_len),
        dtype=torch.long,
        device="cuda",
    )

    attention_mask = torch.ones(
        (1, seq_len),
        dtype=torch.long,
        device="cuda",
    )

    print("Input shape:", tuple(input_ids.shape))

    # ------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------

    if args.attention == "sdpa_flash":
        print(
            "Forcing PyTorch SDPA backend:",
            SDPBackend.FLASH_ATTENTION,
        )

        # Only allow FlashAttention.
        #
        # If PyTorch cannot execute this workload with FlashAttention,
        # it should fail rather than silently falling back to another
        # SDPA implementation.
        with torch.inference_mode(), sdpa_kernel(
            SDPBackend.FLASH_ATTENTION
        ):
            logits = compiled(
                input_ids,
            )

    else:
        with torch.inference_mode():
            logits = compiled(
                input_ids,
                attention_mask,
            )

    torch.cuda.synchronize()

    print("Logits shape:", tuple(logits.shape))
    print("Done")


if __name__ == "__main__":
    main()
