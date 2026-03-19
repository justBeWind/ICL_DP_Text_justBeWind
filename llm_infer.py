import pandas as pd
import json
import models
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer
import re
import argparse
import os
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version_folder", type=str, default='./Noise_version/cm_decode_RoPE_True_eps_3.0_top_20/')
    parser.add_argument("--version", type=str, default='.')
    parser.add_argument("--is_map", action='store_true', help="Enable remapping")
    parser.add_argument("--shot", type=str, choices=['0', '2'], required=True)
    parser.add_argument("--save_folder", type=str, default='./Eval_Final/')
    parser.add_argument("--no_private", action='store_true', help="Set this to use original context (Baseline)")
    return parser


Summarize_Prompt_Tamplete_2_shot = """Context 1: {dialogue1}
Response 1: {summary1}
Context 2: {dialogue2}
Response 2: {summary2}
The above is an example of a summary sentence, please help me summarise the following sentence.
Context:{private_dialogue}
Response: """


Summarize_Prompt_Tamplete_0_shot = """Please help me summarise the following sentence.
Context:{private_dialogue}
Response: """

def replace_words(text, word_map):
    if not word_map:
        return text
    def replace(match):
        word = match.group(0)
        return word_map.get(word, word)
    
    # Sort keys by length descending to match longest possible word first to avoid partial matches
    sorted_keys = sorted(word_map.keys(), key=len, reverse=True)
    if not sorted_keys:
        return text
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in sorted_keys) + r')\b')
    
    return pattern.sub(replace, text)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    with open(os.path.join(args.version_folder,args.version, 'train.json'), 'r', encoding='utf-8') as file:
            train_data = json.load(file)

    with open(os.path.join(args.version_folder,args.version, 'test.json'), 'r', encoding='utf-8') as file:
            test_data = json.load(file)

    # Use Llama-3 tokenizer as the base for mapping logic
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct' 
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    target_model = os.environ.get("DEFAULT_MODEL", "gpt-3.5-turbo")
    llm = models.OpenAILLM(model_path=target_model)
    print(f"\n[Strict Professor Audit]: Connected to Black-box Evaluation LLM: {target_model}\n")
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # Automatic filename generation based on experiment type
    prefix = "Baseline_" if args.no_private else "Private_"
    suffix = "_mapped" if args.is_map else ""
    version_label = args.version_folder.strip('./').replace('/', '_')
    output_filename = os.path.join(args.save_folder, f"{prefix}{args.shot}shot_{version_label}{suffix}.json")
    
    print(f'Starting inference on {len(test_data)} samples...')

    use_private = not args.no_private
    answer_data = []
    
    for item in tqdm(range(len(test_data)), desc="Evaluating Utility"):
        # Select context
        current_context = test_data[item]['private_context'] if use_private else test_data[item]['context']

        if args.shot == '2':
            ctx1 = train_data[item*2]['private_context'] if use_private else train_data[item*2]['context']
            res1 = train_data[item*2]['private_response'] if use_private else train_data[item*2]['response']
            ctx2 = train_data[item*2+1]['private_context'] if use_private else train_data[item*2+1]['context']
            res2 = train_data[item*2+1]['private_response'] if use_private else train_data[item*2+1]['response']
            
            prompt = Summarize_Prompt_Tamplete_2_shot.format(
                dialogue1=ctx1,
                summary1=res1,
                dialogue2=ctx2,
                summary2=res2,
                private_dialogue=current_context)
        else:
            prompt = Summarize_Prompt_Tamplete_0_shot.format(private_dialogue=current_context)
            
        output = llm.generate(prompt=llm.create_conv_prompt(prompt), temperature=0.1, max_tokens=100)
            
        # Robust String-based Remapping
        remap_sentence = output
        if args.is_map:
            str_map = {}
            if 'private_word_map' in test_data[item]:
                for noised_id, original_id in test_data[item]['private_word_map'].items():
                    # Strip leading spaces/formatting from tokens
                    noised_str = tokenizer.decode([int(noised_id)]).strip()
                    original_str = tokenizer.decode([int(original_id)]).strip()
                    if noised_str and len(noised_str) > 1: # Only map meaningful words
                        str_map[noised_str] = original_str
                
                remap_sentence = replace_words(output, str_map)

        answer_data.append({
            "index": item,
            "response": test_data[item]['response'],
            "llm_response": output,
            "noised_remap_llm_response": remap_sentence,
        })
        
        # Incremental save
        if item % 10 == 0 or item == len(test_data) - 1:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(answer_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Inference completed. Results saved to: {output_filename}")
