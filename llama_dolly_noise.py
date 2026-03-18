import torch
# from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
from nltk.corpus import stopwords
import string
import json
from tqdm import tqdm, trange
import os
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from typing import Tuple


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=3)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed",type=int,default=50)
    parser.add_argument("--device",type=str,default='cuda')
    parser.add_argument("--RoPE",type=bool, default=True, required=False)
    parser.add_argument("--combine_method",type=str,choices=['combine', 'decode'], default="decode")
    parser.add_argument("--use_dynamic_k", type=bool, default=True, help="Enable DYNTEXT Dynamic K")
    parser.add_argument("--use_structure", type=bool, default=True, help="Enable DP-ST Structure Extractor")
    return parser


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_shape = xq.shape
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).reshape(orig_shape)
    xk_out = torch.view_as_real(xk_ * freqs_cis).reshape(orig_shape)
    # xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    # xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def load_data(dataset=None):

    with open('./dolly_summarization/train.jsonl', 'r') as f:
        train_data = [json.loads(line) for line in f]

    with open('./dolly_summarization/test.jsonl', 'r') as f:
        test_data = [json.loads(line) for line in f]

    return train_data, test_data

from structure_extractor import StructureExtractor
from dynamic_dp_core import calculate_dynamic_K

def generate_noised_sentence_s2(df, args, type, tokenizer, model):
    print("type:", type)
    vocab = tokenizer.get_vocab()
    word_embeddings_layer = model.get_input_embeddings().to(args.device)

    punct = list(string.punctuation)
    stop_words = set(stopwords.words('english'))
    special_chars = {'\r', '\n'}

    if getattr(args, 'use_structure', True):
        structure_extractor = StructureExtractor(use_openie=False)

    new_dataset = []
    word_embeddings_layer = model.model.get_input_embeddings().to(args.device)

    audit_stats = {
        "total_context_tokens": 0,
        "structural_context_tokens": 0,
        "sum_context_dynamic_k": 0.0,
        "total_response_tokens": 0,
        "structural_response_tokens": 0,
        "sum_response_dynamic_k": 0.0,
    }

    vocab_ids = torch.tensor(list(vocab.values()), device=args.device)

    for index in trange(len(df)):
        private_word_map = {}

        inputs = tokenizer(df[index]['context'], return_tensors='pt').to(args.device)

        input_ids = inputs['input_ids']
        with torch.no_grad():
            word_embeddings = word_embeddings_layer(input_ids)

            # position_ids = torch.arange(0, input_ids.size(1)).squeeze(0).cpu()
            # position_embeddings = model.wpe(position_ids).to(args.device)
        final_embeddings = word_embeddings.squeeze(0)
        # final_embeddings = final_embeddings.squeeze(0)
        if args.RoPE:
            seq_len = input_ids.size(1)
            freqs_cis = precompute_freqs_cis(model.config.hidden_size, seq_len).to(args.device)
            query, key = apply_rotary_emb(word_embeddings, word_embeddings, freqs_cis)
            final_embeddings = query.squeeze(0)

        structural_words = set()
        structural_indices = set()
        if getattr(args, 'use_structure', True):
            text_str = df[index]['context']
            structural_words = structure_extractor.extract_structural_words(text_str)
            for i in range(input_ids.size(1)):
                if tokenizer.decode([input_ids[0, i]]).strip().lower() in structural_words:
                    structural_indices.add(i)

        probabilities_list, top_indices, dynamic_k_list = Dynamic_DP_get_probabilities(
                                        final_embeddings, word_embeddings_layer, vocab_ids, structural_indices, args)
        
        audit_stats["total_context_tokens"] += input_ids.size(1)
        audit_stats["structural_context_tokens"] += len(structural_indices)
        audit_stats["sum_context_dynamic_k"] += float(sum(dynamic_k_list))

        # Randomly select a token to replace the original token based on probability
        new_tokens = []
        if args.combine_method == 'decode':
            for i in range(input_ids.size(1)):
                token_str = tokenizer.decode([input_ids[0, i]]).strip()
                if token_str == tokenizer.bos_token:
                    continue
                if (token_str.lower() in stop_words) or (token_str in string.punctuation) or (token_str in special_chars):
                    new_tokens.append(input_ids[0, i].cpu().item())
                elif int(input_ids[0, i].cpu()) in private_word_map:
                    new_tokens.append(private_word_map[int(input_ids[0, i].cpu())])
                    # print(token_str, "--", private_word_map[token_str])
                else:
                    k_val = dynamic_k_list[i]
                    top_tokens = [int(vocab_ids[idx].cpu()) for idx in top_indices[i][:k_val]]
                    top_probs = probabilities_list[i][0]
                    chosen_token = np.random.choice(top_tokens, p=top_probs)
                    chosen_token = int(chosen_token)
                    private_word_map[int(input_ids[0, i].cpu())] = chosen_token
                    new_tokens.append(chosen_token)
            word_str = tokenizer.decode(new_tokens)
        # firt combine method
        #################################################
        elif args.combine_method == 'combine':
            for i in range(input_ids.size(1)):
                token_str = tokenizer.decode([input_ids[0, i]]).strip()
                if token_str == tokenizer.bos_token:
                    continue
                if (token_str.lower() in stop_words) or (token_str in string.punctuation) or (token_str in special_chars):
                    new_tokens.append(token_str)
                elif token_str in private_word_map:
                    new_tokens.append(private_word_map[token_str])
                    # print(token_str, "--", private_word_map[token_str])
                else:
                    k_val = dynamic_k_list[i]
                    top_tokens = [tokenizer.decode([vocab_ids[idx]]) for idx in top_indices[i][:k_val]]
                    top_probs = probabilities_list[i][0]
                    chosen_token = np.random.choice(top_tokens, p=top_probs).strip()
                    private_word_map[token_str] = chosen_token
                    new_tokens.append(chosen_token)
            new_sentence = " ".join(new_tokens)
            word_str = ' '.join(new_sentence.split())

        # answer perturb
        answer_inputs = tokenizer(df[index]['response'], return_tensors='pt').to(args.device)

        answer_input_ids = answer_inputs['input_ids']
        with torch.no_grad():
            word_embeddings = word_embeddings_layer(answer_input_ids)

            # position_ids = torch.arange(0, input_ids.size(1)).squeeze(0).cpu()
            # position_embeddings = model.wpe(position_ids).to(args.device)
        final_embeddings = word_embeddings.squeeze(0)
        # final_embeddings = final_embeddings.squeeze(0)
        if args.RoPE:
            seq_len = answer_input_ids.size(1)
            freqs_cis = precompute_freqs_cis(model.config.hidden_size, seq_len).to(args.device)
            query, key = apply_rotary_emb(word_embeddings, word_embeddings, freqs_cis)
            final_embeddings = query.squeeze(0)
        
        ans_structural_indices = set()
        if getattr(args, 'use_structure', True):
            ans_text_str = df[index]['response']
            ans_structural_words = structure_extractor.extract_structural_words(ans_text_str)
            for i in range(answer_input_ids.size(1)):
                if tokenizer.decode([answer_input_ids[0, i]]).strip().lower() in ans_structural_words:
                    ans_structural_indices.add(i)

        probabilities_list, top_indices, dynamic_k_list = Dynamic_DP_get_probabilities(
                                        final_embeddings, word_embeddings_layer, vocab_ids, ans_structural_indices, args)
        
        audit_stats["total_response_tokens"] += answer_input_ids.size(1)
        audit_stats["structural_response_tokens"] += len(ans_structural_indices)
        audit_stats["sum_response_dynamic_k"] += float(sum(dynamic_k_list))

        new_tokens = []
        if args.combine_method == 'decode':
            for i in range(answer_input_ids.size(1)):
                token_str = tokenizer.decode([answer_input_ids[0, i]]).strip()
                if token_str == tokenizer.bos_token:
                    continue
                if (token_str.lower() in stop_words) or (token_str in string.punctuation) or (token_str in special_chars):
                    new_tokens.append(answer_input_ids[0, i].cpu().item())
                elif int(answer_input_ids[0, i].cpu()) in private_word_map:
                    new_tokens.append(private_word_map[int(answer_input_ids[0, i].cpu())])
                    # print(token_str, "--", private_word_map[token_str])
                else:
                    k_val = dynamic_k_list[i]
                    top_tokens = [int(vocab_ids[idx].cpu()) for idx in top_indices[i][:k_val]]
                    top_probs = probabilities_list[i][0]
                    chosen_token = np.random.choice(top_tokens, p=top_probs)
                    chosen_token = int(chosen_token)
                    private_word_map[int(answer_input_ids[0, i].cpu())] = chosen_token
                    new_tokens.append(chosen_token)
            answer_word_str = tokenizer.decode(new_tokens)
        # firt combine method
        #################################################
        elif args.combine_method == 'combine':
            for i in range(answer_input_ids.size(1)):
                token_str = tokenizer.decode([answer_input_ids[0, i]]).strip()
                if token_str == tokenizer.bos_token:
                    continue
                if (token_str.lower() in stop_words) or (token_str in string.punctuation) or (token_str in special_chars):
                    new_tokens.append(token_str)
                elif token_str in private_word_map:
                    new_tokens.append(private_word_map[token_str])
                    # print(token_str, "--", private_word_map[token_str])
                else:
                    k_val = dynamic_k_list[i]
                    top_tokens = [tokenizer.decode([vocab_ids[idx]]) for idx in top_indices[i][:k_val]]
                    top_probs = probabilities_list[i][0]
                    chosen_token = np.random.choice(top_tokens, p=top_probs).strip()
                    private_word_map[token_str] = chosen_token
                    new_tokens.append(chosen_token)
            new_sentence = " ".join(new_tokens)
            answer_word_str = ' '.join(new_sentence.split())

        
        new_dataset.append({
            "context": df[index]['context'],
            "response": df[index]['response'],
            "private_context": word_str,
            "private_response": answer_word_str,
            "private_word_map": private_word_map
        })
    if not os.path.exists(f"Noise_version/cm_{args.combine_method}_RoPE_{args.RoPE}_eps_{args.eps}_top_{args.top_k}"):
        os.makedirs(f"Noise_version/cm_{args.combine_method}_RoPE_{args.RoPE}_eps_{args.eps}_top_{args.top_k}")
    
    with open(f"Noise_version/cm_{args.combine_method}_RoPE_{args.RoPE}_eps_{args.eps}_top_{args.top_k}/"+ type +".json", 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=4)
        
    audit_stats["avg_context_dynamic_k"] = round(audit_stats["sum_context_dynamic_k"] / max(1, audit_stats["total_context_tokens"]), 2)
    audit_stats["avg_response_dynamic_k"] = round(audit_stats["sum_response_dynamic_k"] / max(1, audit_stats["total_response_tokens"]), 2)
    with open(f"Noise_version/cm_{args.combine_method}_RoPE_{args.RoPE}_eps_{args.eps}_top_{args.top_k}/"+ type +"_audit.json", 'w', encoding='utf-8') as f:
        json.dump(audit_stats, f, ensure_ascii=False, indent=4)
      
def Dynamic_DP_get_probabilities(final_embeddings, word_embeddings_layer, vocab_ids, structural_indices, args, batch_size = 10000):
    final_norm = final_embeddings / final_embeddings.norm(dim=1, keepdim=True)
    num_batches = (len(vocab_ids) + batch_size - 1) // batch_size
    similarities = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(vocab_ids))
        batch_ids = vocab_ids[start_idx:end_idx]
        with torch.no_grad():
            batch_embeddings = word_embeddings_layer(batch_ids).detach()

        batch_norm = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
        batch_similarities = torch.matmul(final_norm, batch_norm.T)
        similarities.append(batch_similarities)
    similarities = torch.cat(similarities, dim=1)
    
    if getattr(args, 'use_dynamic_k', True):
        dynamic_k_list = calculate_dynamic_K(similarities, args.top_k).cpu().numpy()
    else:
        dynamic_k_list = [args.top_k] * similarities.size(0)
        
    top_similarities, top_indices = similarities.topk(args.top_k, dim=1, largest=True, sorted=True)
    min_values, _ = torch.min(top_similarities, dim=1, keepdim=True)
    max_values, _ = torch.max(top_similarities, dim=1, keepdim=True)
    
    diff = max_values - min_values
    diff[diff == 0] = 1e-8
    normalized_top_similarities = (top_similarities - min_values) / diff
    
    probabilities_list = []
    for i in range(similarities.size(0)):
        is_structural = (i in structural_indices)
        if getattr(args, 'use_structure', True):
            if is_structural:
                current_eps = args.eps # Strict protection for structural backbone
            else:
                current_eps = getattr(args, 'background_eps', args.eps * 5.0) # Relaxed privacy for grammatical glue words
        else:
            current_eps = args.eps
            
        k_val = dynamic_k_list[i]
        k_sims = normalized_top_similarities[i, :k_val]
        
        dp_sims = torch.exp(current_eps * k_sims / 2.0)
        probs = torch.softmax(dp_sims, dim=-1).cpu().numpy()
        probabilities_list.append((probs, k_val))
        
    return probabilities_list, top_indices.cpu().numpy(), dynamic_k_list

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_data,test_data = load_data()

    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'  
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)

    # model.to(args.device)

    
    # generate_noised_sentence_s2(df=dev_data, args=args, type='dev', tokenizer=tokenizer, model=model)
    generate_noised_sentence_s2(df=test_data, args=args, type='test', tokenizer=tokenizer, model=model)
    generate_noised_sentence_s2(df=train_data, args=args, type='train', tokenizer=tokenizer, model=model)
    
