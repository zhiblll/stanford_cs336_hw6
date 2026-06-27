from __future__ import annotations

import os
from typing import Any, Callable, Literal
import random
import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from cs336_alignment import grpo
import re
import json
import torch.nn.functional as F

class Mini_Dataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    return grpo.tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    return grpo.get_response_log_probs(model, input_ids, labels, return_token_entropy)


def run_compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    return grpo.compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)


def run_compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    return grpo.compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)


def run_compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    return grpo.compute_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs, importance_reweighting_method, old_log_probs, cliprange, response_mask)


def run_aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    return  grpo.aggregate_loss_across_microbatch(per_token_policy_gradient_loss, mask, loss_normalization, normalization_constant)


def run_grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    return grpo.grpo_train_step(model, 
                                tokenizer, 
                                optimizer, 
                                gradient_accumulation_steps, 
                                max_grad_norm, 
                                reward_fn, 
                                repeated_prompts, 
                                rollout_responses,
                                repeated_ground_truths,
                                group_size,
                                baseline,
                                advantage_eps,
                                advantage_normalizer,
                                importance_reweighting_method,
                                old_log_probs,
                                cliprange,
                                loss_normalization,
                                normalization_constant)


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    a = tensor * mask
    a = a.sum(dim = dim)
    a = a/normalize_constant
    return a


def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    masked_policy_log_probs = response_mask * policy_log_probs
    loss =  - masked_policy_log_probs.sum()/(gradient_accumulation_steps * normalize_constant * policy_log_probs.shape[0])
    loss.backward()
    return loss, {}

def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    records = []
    all_tokens = []
    examples = []
    with open("cs336_alignment/prompts/alpaca_sft.prompt") as f:
        template = f.read().rstrip()
    with open(dataset_path) as f:
        for line in f:
            record = (json.loads(line))
            records.append(record)
    
    if shuffle:
            random.shuffle(records)

    for record in records:        
        prompt = record["prompt"]
        response = record["response"]
        

        text = template.format(instruction=prompt,
                                response=response,)
        ids = tokenizer(text).input_ids
        ids.append(tokenizer.eos_token_id)
        all_tokens.extend(ids)

    for i in range(0, len(all_tokens), seq_length):
        if i + seq_length + 1 > len(all_tokens):
            break
        else:
            tokens = all_tokens[i : i + seq_length + 1]
        
        example = {"input_ids": torch.tensor(tokens[:-1]),
                   "labels": torch.tensor(tokens[1:])}
        
        examples.append(example)

    return Mini_Dataset(examples)

    
        


        
        


            
    

def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    batch_len = len(dataset)
    indice = [i for i in range(batch_len)]
    if shuffle:
        indice = torch.randperm(len(dataset)).tolist()
    batches = []
    for start in range(0, len(indice), batch_size):
        batch_indices = indice[start:start + batch_size]
        batch = {"input_ids": torch.stack([dataset[i]["input_ids"] for i in batch_indices], dim = 0),
                 "labels": torch.stack([dataset[i]["labels"] for i in batch_indices], dim = 0)}

        batches.append(batch)

    return batches

def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    text = model_output.upper()
    match = re.search(r"\b([ABCD])\b", text)

    if match:
        return match.group(1)
    
    return None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    text = model_output.upper()
    match = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)

    if match:
        return match[-1]
    
    return None


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """

    # toknized_prompt_response_chosen = grpo.tokenize_prompt_and_output([prompt], [response_chosen], tokenizer)
    # toknized_prompt_response_chosen_ids = toknized_prompt_response_chosen["input_ids"]
    # toknized_prompt_response_chosen_labels = toknized_prompt_response_chosen["labels"]
    # toknized_prompt_response_chosen_mask = toknized_prompt_response_chosen["response_mask"]
    
    # tokenized_prompt_response_rej = grpo.tokenize_prompt_and_output([prompt], [response_rejected], tokenizer)
    # tokenized_prompt_response_rej_ids = tokenized_prompt_response_rej["input_ids"]
    # toknized_prompt_response_rej_labels = tokenized_prompt_response_rej["labels"]
    # tokenized_prompt_response_rej_mask = tokenized_prompt_response_rej["response_mask"]

    # ref_chosen_log_probs = grpo.get_response_log_probs(lm_ref, toknized_prompt_response_chosen_ids, toknized_prompt_response_chosen_labels)
    # ref_chosen_log_probs =ref_chosen_log_probs["log_probs"]
    # ref_chosen_log_probs = ref_chosen_log_probs * toknized_prompt_response_chosen_mask

    # ref_rej_log_probs = grpo.get_response_log_probs(lm_ref, tokenized_prompt_response_rej_ids, toknized_prompt_response_rej_labels)
    # ref_rej_log_probs = ref_rej_log_probs["log_probs"]
    # ref_rej_log_probs = ref_rej_log_probs * tokenized_prompt_response_rej_mask
    
    # chosen_log_probs = grpo.get_response_log_probs(lm, toknized_prompt_response_chosen_ids, toknized_prompt_response_chosen_labels)
    # chosen_log_probs = chosen_log_probs["log_probs"]
    # chosen_log_probs = chosen_log_probs *  toknized_prompt_response_chosen_mask

    # rej_log_probs = grpo.get_response_log_probs(lm, tokenized_prompt_response_rej_ids, toknized_prompt_response_rej_labels)
    # rej_log_probs = rej_log_probs["log_probs"]
    # rej_log_probs = rej_log_probs * tokenized_prompt_response_rej_mask

    device = next(lm.parameters()).device


    with open("cs336_alignment/prompts/alpaca_sft.prompt") as f:
        template = f.read().rstrip()
        chosen_text = template.format(instruction = prompt, response = response_chosen)
        rej_text = template.format(instruction = prompt, response = response_rejected)
    
    chosen_ids = tokenizer(chosen_text).input_ids
    chosen_ids.append(tokenizer.eos_token_id)

    rej_text_ids = tokenizer(rej_text).input_ids
    rej_text_ids.append(tokenizer.eos_token_id)


    chosen_ids = torch.tensor(chosen_ids, dtype=torch.long, device=device).unsqueeze(0)
    rej_ids = torch.tensor(rej_text_ids, dtype=torch.long, device=device).unsqueeze(0)


    ref_chosen_log_probs = grpo.get_response_log_probs(lm_ref, chosen_ids[:, :-1], chosen_ids[:, 1:])
    ref_chosen_log_probs =ref_chosen_log_probs["log_probs"]
    

    ref_rej_log_probs = grpo.get_response_log_probs(lm_ref, rej_ids[:, :-1], rej_ids[:, 1:])
    ref_rej_log_probs = ref_rej_log_probs["log_probs"]
   
    chosen_log_probs = grpo.get_response_log_probs(lm, chosen_ids[:, :-1], chosen_ids[:, 1:])
    chosen_log_probs = chosen_log_probs["log_probs"]
    

    rej_log_probs = grpo.get_response_log_probs(lm, rej_ids[:, :-1], rej_ids[:, 1:])
    rej_log_probs = rej_log_probs["log_probs"]
    
    loss = (chosen_log_probs - ref_chosen_log_probs) - (rej_log_probs - ref_rej_log_probs)
    loss = loss.sum()
    loss = -F.logsigmoid(loss*beta)

    return loss

   

   

    



    





