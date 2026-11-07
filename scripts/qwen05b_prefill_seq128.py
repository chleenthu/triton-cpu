import argparse
import os

os.environ.setdefault("TRITON_CPU_TARGET", "native")

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.config import CompilationConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        choices=["CPU_ATTN", "TRITON_ATTN"],
        help="omit to let vLLM pick its CPU default",
    )
    args = parser.parse_args()

    print("Model      :", MODEL_ID)
    print("Seq len    :", args.seq_len)
    print("Max tokens :", args.max_tokens)
    print("Attention   :", args.attention_backend)

    compile_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.NONE,
        inductor_compile_config={"cpu_backend": "triton"},
    )

    llm_kwargs = {}
    if args.attention_backend is not None:
        llm_kwargs["attention_backend"] = args.attention_backend

    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        tensor_parallel_size=1,
        compilation_config=compile_config,
        max_model_len=1024,
        max_num_seqs=1,
        **llm_kwargs,
    )

    token_ids = [
        1000 + (i % 10000)
        for i in range(args.seq_len)
    ]

    prompt = TokensPrompt(
        prompt_token_ids=token_ids,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(
        [prompt],
        sampling_params,
        use_tqdm=False,
    )

    output = outputs[0]

    print("Prompt tokens   :", len(output.prompt_token_ids))
    print("Generated tokens:", len(output.outputs[0].token_ids))
    print("Done")


if __name__ == "__main__":
    main()
