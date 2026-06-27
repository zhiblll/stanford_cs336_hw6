# cs336_alignment/grpo.py
import torch
from transformers import PreTrainedTokenizer
import torch.nn.functional as F
from transformers import PreTrainedModel
from torch.nn.utils.rnn import pad_sequence
import math

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer = PreTrainedTokenizer):
    inputs_id_list = []
    labels_id_list = []
    
    full_id_list = []
    
    prompts_len = []
    labels_len = []

   

    for prompt_str, output_str in zip(prompt_strs, output_strs):
        prompt_token = tokenizer(prompt_str, padding = False).input_ids
        output_token = tokenizer(output_str, padding = False).input_ids

        prompt_token = torch.tensor(prompt_token)
        prompts_len.append(len(prompt_token))
        
        output_token = torch.tensor(output_token)
        labels_len.append(len(output_token))
     
        
        inputs_id_list.append(prompt_token)
        labels_id_list.append(output_token)

        full_token = torch.cat((prompt_token, output_token), dim = 0)
        
        full_id_list.append(full_token)

    full_id_list = pad_sequence(full_id_list, batch_first=True, padding_value=tokenizer.pad_token_id)

    full_id_len = len(full_id_list[0])
    inputs_ids= [ids[:-1] for ids in full_id_list]
    labels_ids = [ids[1:] for ids in full_id_list]


    input_ids = torch.stack(inputs_ids, dim=0)
    labels = torch.stack(labels_ids, dim=0)
    mask_id_list = [[0] * (prompt_len - 1) + [1] * (label_len) + [0] * (full_id_len - label_len - prompt_len) for prompt_len, label_len in zip(prompts_len, labels_len)]
    mask_id_list = torch.tensor(mask_id_list)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": mask_id_list,
    }


def get_response_log_probs(model: PreTrainedModel, input_ids, labels, return_token_entropy=True):
    
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim = -1, index = labels.unsqueeze(-1)).squeeze(-1)
    
    if return_token_entropy:
        probs = F.softmax(logits, dim=-1)
        token_entropy = -(probs * log_probs).sum(dim=-1)
        
    out = {"log_probs": token_log_probs, "token_entropy": token_entropy} if return_token_entropy else {"log_probs": token_log_probs}
    return out

def compute_rollout_rewards(reward_fn,
    rollout_responses,
    repeated_ground_truths,):
    # Compute reward for the response using the reward model
    reward_dicts = [reward_fn(response, ground_truth) for response, ground_truth in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([reward_dict["reward"] for reward_dict in reward_dicts], dtype=torch.float)
    format_rewards = torch.tensor(
        [x["format_reward"] for x in reward_dicts],
        dtype=torch.float32,
    )

    answer_rewards = torch.tensor(
        [x["answer_reward"] for x in reward_dicts],
        dtype=torch.float32,
    )

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "format_reward_mean": format_rewards.mean().item(),
        "answer_reward_mean": answer_rewards.mean().item(),
    }

    return raw_rewards, metadata

    

def compute_group_normalized_rewards(raw_rewards, group_size=2, baseline="mean", advantage_eps = 1e-6, advantage_normalizer = "std"):
    reward_group = raw_rewards.view(-1, group_size)  # Assuming group_size responses per prompt
    group_mean = reward_group.mean(dim=1, keepdim=True)
    group_std = reward_group.std(dim = -1, keepdim=True) 
    if baseline == "mean":
        centered_rewards = reward_group - group_mean

    elif  baseline == "none":
        centered_rewards = reward_group
    
    if advantage_normalizer == "std":
        advantages = centered_rewards / (group_std + advantage_eps)  # Add epsilon to avoid division by zero

    elif advantage_normalizer == "none":
        advantages = centered_rewards
    elif advantage_normalizer == "mean":
        advantages = centered_rewards / (group_mean + advantage_eps)  # Add epsilon to avoid division by zero
    
    advantages = advantages.view(-1)  # Flatten back to original shape
    
    
    metadata = {
        "group_mean": group_mean.mean().item(),
        "group_std": group_std.mean().item(),
        "normalized_rewards": advantages}
    return advantages, metadata




   

def compute_policy_gradient_loss(
    raw_rewards_or_advantages,
    policy_log_probs,
    importance_reweighting_method = "none",
    old_log_probs = None,
    cliprange = None,
    response_mask=None):
    # Compute the policy gradient loss
    advantages = raw_rewards_or_advantages
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)  # Ensure advantages has the same shape as policy_log_probs
    
    if importance_reweighting_method == "none":
        
        policy_loss = -(advantages * policy_log_probs)
        return policy_loss, {}
    
    elif importance_reweighting_method == "noclip":
        
        importance_weight = torch.exp(policy_log_probs - old_log_probs)
        policy_loss = - importance_weight * advantages
        metadata = {
            "importance_weight_mean": importance_weight.mean().item(),
            "importance_weight_std": importance_weight.std().item(),
            "importance_weight_min": importance_weight.min().item(),
            "importance_weight_max": importance_weight.max().item(),    
                        }
        return policy_loss, metadata
        

    
    elif importance_reweighting_method == "grpo":
        importance_weight = torch.exp(policy_log_probs - old_log_probs)
        unclipped_obj = importance_weight * advantages
        clipped_importance_weight = torch.clamp(importance_weight, 1 - cliprange, 1 + cliprange)
        clipped_obj = clipped_importance_weight * advantages
        
        obj = torch.min(unclipped_obj, clipped_obj)
        policy_loss = -obj

        metadata = {"clip_fraction": (importance_weight > 1 + cliprange).float().mean().item() + (importance_weight < 1 - cliprange).float().mean().item(),}
        return policy_loss, metadata
        
    
    elif importance_reweighting_method == "gspo":
        importante_weight_log = policy_log_probs - old_log_probs
        mask = response_mask.to(importante_weight_log.device)
        seq_log_weight = (importante_weight_log * mask).sum(dim = 1, keepdim=True) / (mask.sum(dim = 1, keepdim=True).clamp_min(1))
        seq_weight = torch.exp(seq_log_weight) 
        clipped_importance_weight = torch.clamp(seq_weight, 1 - cliprange, 1 + cliprange)
        clipped_obj = clipped_importance_weight * advantages
        obj = torch.min(seq_weight * advantages, clipped_obj)
        policy_loss = - obj
        policy_loss = policy_loss * torch.ones_like(policy_log_probs)
        metadata = {"clip_fraction": (seq_weight > 1 + cliprange).float().mean().item() + (seq_weight < 1 - cliprange).float().mean().item(),}
        return policy_loss, metadata



    


    
    return policy_loss

    

def aggregate_loss_across_microbatch(per_token_policy_gradient_loss,
    mask,
    loss_normalization = "sequence",
    normalization_constant = None):
    # Aggregate the policy loss across microbatches
    mask = mask.to(per_token_policy_gradient_loss.device)
    masked_loss  = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        loss = masked_loss.sum(dim=-1) / torch.clamp(mask.sum(dim=-1), min=1)
        loss = loss.mean()
    elif loss_normalization == "constant":
        loss = masked_loss.sum()/normalization_constant
    
    return loss

def grpo_train_step(
        model,
        tokenizer,
        optimizer,
        gradient_accumulation_steps,
        max_grad_norm,
        reward_fn,
        repeated_prompts,
        rollout_responses,
        repeated_ground_truths,
        group_size,
        baseline = "mean",
        advantage_eps = 1e-6,
        advantage_normalizer = "std",
        importance_reweighting_method = "none",
        old_log_probs = None,
        cliprange = None,
        loss_normalization = "sequence",
        normalization_constant=None,
):
    device = next(model.parameters()).device
    optimizer.zero_grad()
    metadata ={}
    raw_reward, raw_reward_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    raw_reward = raw_reward.to(device)
    advantages, advtanges_metadata = compute_group_normalized_rewards(raw_reward, group_size, baseline, advantage_eps, advantage_normalizer)
    advantages = advantages.to(device)
    metadata.update(raw_reward_metadata)
    metadata.update(advtanges_metadata)
    if torch.all(advantages == 0):
        metadata["skipped_update"] = True
        return torch.zeros((), device=device), metadata

    toknized_ids = tokenize_prompt_and_output(repeated_prompts, rollout_responses,tokenizer)

    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(device)

    input_ids = toknized_ids["input_ids"]
    input_ids = input_ids.to(device)
    label_ids = toknized_ids["labels"]
    label_ids = label_ids.to(device)
    response_mask = toknized_ids["response_mask"]
    response_mask = response_mask.to(device)

    micro_batch_size = math.ceil(input_ids.shape[0]/gradient_accumulation_steps)
    total_loss = 0.0
    micro_batch_mata = {}
    for step in range(gradient_accumulation_steps):
        start = step*micro_batch_size
        end = (step+1)*micro_batch_size
        end = min(end, input_ids.shape[0])

        log_probs_dict = get_response_log_probs(model, input_ids[start:end], label_ids[start:end], return_token_entropy=False)
        log_probs = log_probs_dict["log_probs"]
        if  old_log_probs is not None:
            old_log_probs_microbatch = old_log_probs[start:end]
        else:
            old_log_probs_microbatch = None

        raw_loss, micro_batch_policy_metadata = compute_policy_gradient_loss(advantages[start:end], log_probs, importance_reweighting_method, old_log_probs_microbatch, cliprange, response_mask[start:end])
        loss = aggregate_loss_across_microbatch(raw_loss, response_mask[start:end], loss_normalization, normalization_constant)
        for key, values in micro_batch_policy_metadata.items():
            if key not in micro_batch_mata:
                micro_batch_mata[key] = []
            micro_batch_mata[key].append(values)
        if loss_normalization == "sequence":
            loss = loss * (1/gradient_accumulation_steps)
            
        loss.backward()
        total_loss += loss
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    metadata.update(micro_batch_mata)
    
    return total_loss.detach(), metadata



        
