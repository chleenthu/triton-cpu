import argparse

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
        default="TRITON_ATTN",
        choices=["TRITON_ATTN", "FLASH_ATTN"],
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

    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        tensor_parallel_size=1,

        # Force a Triton attention implementation so attention
        # also appears in the Triton workload we analyze.
        attention_backend=args.attention_backend,

        # vLLM custom compile path, but disable CUDA Graphs
        # to keep the compiler experiment easier to interpret.
        compilation_config=compile_config,

        max_model_len=1024,
        max_num_seqs=1,
        gpu_memory_utilization=0.70,
    )

    # Fixed-length tokenized input.
    # All IDs are safely inside Qwen's vocabulary.
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
