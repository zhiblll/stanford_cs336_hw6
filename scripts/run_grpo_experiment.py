from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment import grpo
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer


PROMPT_FILES = {
    "question_only": "question_only.prompt",
    "r1_zero": "r1_zero.prompt",
    "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
}


VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "standard": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
        "importance_reweighting_method": "none",
        "cliprange": None,
    },
    "grpo_constant": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "constant",
        "importance_reweighting_method": "none",
        "cliprange": None,
    },
    "dr_grpo": {
        "baseline": "mean",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
        "importance_reweighting_method": "none",
        "cliprange": None,
    },
    "rft": {
        "baseline": "none",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
        "importance_reweighting_method": "none",
        "cliprange": None,
    },
    "maxrl": {
        "baseline": "none",
        "advantage_normalizer": "mean",
        "loss_normalization": "constant",
        "importance_reweighting_method": "none",
        "cliprange": None,
    },
    "offpolicy_naive": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
        "importance_reweighting_method": "none",
        "cliprange": None,
    },
    "offpolicy_noclip": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
        "importance_reweighting_method": "noclip",
        "cliprange": None,
    },
    "offpolicy_clip": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
        "importance_reweighting_method": "grpo",
        "cliprange": 0.2,
    },
    "offpolicy_gspo": {
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "sequence",
        "importance_reweighting_method": "gspo",
        "cliprange": 3e-4,
    },
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def extract_gsm8k_answer(answer: str) -> str:
    return answer.split("####")[-1].strip() if "####" in answer else answer.strip()


def load_prompt_template(prompt_name: str) -> str:
    prompt_path = Path("cs336_alignment/prompts") / PROMPT_FILES[prompt_name]
    return prompt_path.read_text(encoding="utf-8").rstrip()


def format_prompt(template: str, example: dict[str, Any]) -> str:
    question = example.get("question") or example.get("prompt") or example.get("instruction")
    if question is None:
        raise KeyError("Example must contain one of: question, prompt, instruction.")
    return template.format(question=question, instruction=question)


def get_ground_truth(example: dict[str, Any]) -> str:
    if "answer" in example:
        return extract_gsm8k_answer(str(example["answer"]))
    if "ground_truth" in example:
        return str(example["ground_truth"]).strip()
    if "target" in example:
        return str(example["target"]).strip()
    raise KeyError("Example must contain one of: answer, ground_truth, target.")


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().float().mean().cpu().item()
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_reward_fn(prompt_name: str) -> Callable[[str, str], dict[str, float]]:
    if prompt_name == "question_only":
        return question_only_reward_fn
    return r1_zero_reward_fn


def make_sampling_params(args: argparse.Namespace, seed: int, n: int) -> dict[str, Any]:
    params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_generation_tokens,
        "n": n,
        "seed": seed,
    }
    if args.prompt_name in {"r1_zero", "r1_zero_three_shot"}:
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    return params


@torch.no_grad()
def generate_hf(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    args: argparse.Namespace,
    seed: int,
) -> list[str]:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    outputs: list[str] = []
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for start in range(0, len(prompts), args.generation_batch_size):
        prompt_batch = prompts[start : start + args.generation_batch_size]
        encoded = tokenizer(prompt_batch, return_tensors="pt", padding=True).to(device)
        input_width = encoded["input_ids"].shape[1]
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_generation_tokens,
            num_return_sequences=args.group_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            remove_invalid_values=True,
            renormalize_logits=True,
        )
        for sequence in generated:
            outputs.append(tokenizer.decode(sequence[input_width:], skip_special_tokens=True))
        del encoded, generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if was_training:
        model.train()
    return outputs


def generate_rollouts(
    model: torch.nn.Module,
    tokenizer,
    vllm_server: VLLMServer | None,
    prompts: list[str],
    args: argparse.Namespace,
    seed: int,
) -> list[str]:
    if args.generation_backend == "hf":
        return generate_hf(model, tokenizer, prompts, args, seed)

    assert vllm_server is not None
    sampling_params = make_sampling_params(args, seed=seed, n=args.group_size)
    completions = vllm_server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=args.generation_batch_size,
    )
    return [completion.text for completion in completions]


@torch.no_grad()
def compute_old_log_probs(
    model: torch.nn.Module,
    tokenizer,
    repeated_prompts: list[str],
    rollout_responses: list[str],
    batch_size: int,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(repeated_prompts), batch_size):
        prompt_chunk = repeated_prompts[start : start + batch_size]
        response_chunk = rollout_responses[start : start + batch_size]
        tokenized = grpo.tokenize_prompt_and_output(prompt_chunk, response_chunk, tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        labels = tokenized["labels"].to(device)
        old = grpo.get_response_log_probs(model, input_ids, labels, return_token_entropy=False)["log_probs"]
        chunks.append(old.cpu())
    if was_training:
        model.train()
    return torch.cat(chunks, dim=0)


def chunk_rollouts(
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    train_batch_size: int,
    group_size: int,
) -> list[tuple[list[str], list[str], list[str]]]:
    if train_batch_size % group_size != 0:
        raise ValueError("train_batch_size must be a multiple of group_size to preserve reward groups.")
    chunks = []
    for start in range(0, len(rollout_responses), train_batch_size):
        end = min(start + train_batch_size, len(rollout_responses))
        if end - start < group_size:
            continue
        chunks.append(
            (
                repeated_prompts[start:end],
                rollout_responses[start:end],
                repeated_ground_truths[start:end],
            )
        )
    return chunks


def evaluate(
    model: torch.nn.Module,
    tokenizer,
    vllm_server: VLLMServer | None,
    examples: list[dict[str, Any]],
    prompt_template: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    args: argparse.Namespace,
    step: int,
) -> dict[str, Any]:
    if not examples:
        return {}
    eval_examples = examples[: args.eval_examples]
    prompts = [format_prompt(prompt_template, example) for example in eval_examples]
    ground_truths = [get_ground_truth(example) for example in eval_examples]
    eval_args = argparse.Namespace(**vars(args))
    eval_args.group_size = 1
    responses = generate_rollouts(
        model=model,
        tokenizer=tokenizer,
        vllm_server=vllm_server,
        prompts=prompts,
        args=eval_args,
        seed=args.seed + 10_000 + step,
    )
    rewards = [reward_fn(response, gt) for response, gt in zip(responses, ground_truths)]
    response_lengths = [len(tokenizer(response).input_ids) for response in responses]
    return {
        "eval/reward": sum(r["reward"] for r in rewards) / len(rewards),
        "eval/format_reward": sum(r["format_reward"] for r in rewards) / len(rewards),
        "eval/answer_reward": sum(r["answer_reward"] for r in rewards) / len(rewards),
        "eval/avg_response_tokens": sum(response_lengths) / len(response_lengths),
        "eval/examples": [
            {
                "prompt": prompts[i],
                "response": responses[i],
                "ground_truth": ground_truths[i],
                "reward": rewards[i],
            }
            for i in range(min(args.log_generations, len(responses)))
        ],
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    variant = VARIANT_CONFIGS[args.variant].copy()
    if args.cliprange is not None:
        variant["cliprange"] = args.cliprange

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    device = args.device
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16 if args.dtype == "fp16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_examples = load_jsonl(args.train_path)
    val_examples = load_jsonl(args.val_path) if args.val_path else []
    prompt_template = load_prompt_template(args.prompt_name)
    reward_fn = get_reward_fn(args.prompt_name)

    vllm_server = None
    if args.generation_backend == "vllm":
        vllm_server = VLLMServer(
            model_id=args.model,
            port=args.vllm_port,
            gpu=args.vllm_gpu,
            seed=args.seed,
            launch_server=not args.no_launch_vllm,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
        vllm_server.start()
        vllm_server.init_weight_sync(policy_device=device)

    n_prompts = args.rollout_batch_size // args.group_size
    if n_prompts <= 0:
        raise ValueError("rollout_batch_size must be >= group_size.")

    for step in range(args.steps):
        start_time = time.time()
        sampled = random.sample(train_examples, k=min(n_prompts, len(train_examples)))
        if len(sampled) < n_prompts:
            sampled = [random.choice(train_examples) for _ in range(n_prompts)]

        prompts = [format_prompt(prompt_template, example) for example in sampled]
        ground_truths = [get_ground_truth(example) for example in sampled]

        if vllm_server is not None:
            vllm_server.sync_policy_weights(model)

        rollout_responses = generate_rollouts(
            model=model,
            tokenizer=tokenizer,
            vllm_server=vllm_server,
            prompts=prompts,
            args=args,
            seed=args.seed + step,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        repeated_prompts = [prompt for prompt in prompts for _ in range(args.group_size)]
        repeated_ground_truths = [gt for gt in ground_truths for _ in range(args.group_size)]

        if len(rollout_responses) != len(repeated_prompts):
            raise RuntimeError(
                f"Expected {len(repeated_prompts)} rollouts, got {len(rollout_responses)}."
            )

        rollout_chunks = chunk_rollouts(
            repeated_prompts=repeated_prompts,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            train_batch_size=args.train_batch_size,
            group_size=args.group_size,
        )
        if not rollout_chunks:
            raise RuntimeError("No train chunks were created. Check train_batch_size and group_size.")

        old_log_prob_chunks: list[torch.Tensor | None] = []
        if variant["importance_reweighting_method"] != "none":
            for prompt_chunk, response_chunk, _ in rollout_chunks:
                old_log_prob_chunks.append(
                    compute_old_log_probs(
                        model=model,
                        tokenizer=tokenizer,
                        repeated_prompts=prompt_chunk,
                        rollout_responses=response_chunk,
                        batch_size=args.train_batch_size,
                    )
                )
        else:
            old_log_prob_chunks = [None for _ in rollout_chunks]

        step_losses = []
        step_metadata: dict[str, Any] = {}
        for chunk_idx, ((prompt_chunk, response_chunk, gt_chunk), old_log_probs) in enumerate(
            zip(rollout_chunks, old_log_prob_chunks)
        ):
            normalization_constant = args.normalization_constant
            if normalization_constant is None and variant["loss_normalization"] == "constant":
                normalization_constant = len(response_chunk) * args.max_generation_tokens

            loss, metadata = grpo.grpo_train_step(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                reward_fn=reward_fn,
                repeated_prompts=prompt_chunk,
                rollout_responses=response_chunk,
                repeated_ground_truths=gt_chunk,
                group_size=args.group_size,
                baseline=variant["baseline"],
                advantage_eps=args.advantage_eps,
                advantage_normalizer=variant["advantage_normalizer"],
                importance_reweighting_method=variant["importance_reweighting_method"],
                old_log_probs=old_log_probs,
                cliprange=variant["cliprange"],
                loss_normalization=variant["loss_normalization"],
                normalization_constant=normalization_constant,
            )
            step_losses.append(float(loss.detach().cpu()))
            for key, value in metadata.items():
                step_metadata[f"train/{key}"] = jsonable(value)
            step_metadata["train/chunks"] = chunk_idx + 1

        row: dict[str, Any] = {
            "step": step,
            "variant": args.variant,
            "prompt_name": args.prompt_name,
            "train/loss": sum(step_losses) / len(step_losses),
            "train/rollout_batch_size": len(rollout_responses),
            "train/elapsed_sec": time.time() - start_time,
            **step_metadata,
        }

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps - 1):
            row.update(evaluate(model, tokenizer, vllm_server, val_examples, prompt_template, reward_fn, args, step))

        append_jsonl(metrics_path, row)
        print(json.dumps(jsonable(row), ensure_ascii=False))

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            ckpt_dir = output_dir / f"checkpoint_step_{step + 1}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reduced or full CS336 GRPO experiments.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-name", choices=sorted(PROMPT_FILES), default="r1_zero")
    parser.add_argument("--variant", choices=sorted(VARIANT_CONFIGS), default="standard")
    parser.add_argument("--generation-backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--normalization-constant", type=float, default=None)
    parser.add_argument("--cliprange", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-generation-tokens", type=int, default=512)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--log-generations", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--no-launch-vllm", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
