# import evaluate
import pandas as pd
import json
import models
from fastchat.model import get_conversation_template
# from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoTokenizer
import re
import argparse
import os
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version_folder", type=str, default='./privatized_dataset/glove.42B.300d/conservative/')
    parser.add_argument("--version", type=str, default='eps_1.0_top_20_s1_save_stop_words_True')
    parser.add_argument("--is_map", type=bool, default=False, required=False)
    parser.add_argument("--shot",type=str,choices=['0', '2'])
    parser.add_argument("--save_folder",type=str, default='./Eval_Final/')
    parser.add_argument("--use_private", type=bool, default=True, help="Whether to use noised context for inference")
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

def create_prompt(prompt,model_name):

    conv_template = get_conversation_template(
        model_name
    )
    conv_template.append_message(conv_template.roles[0], prompt)
    conv_template.append_message(conv_template.roles[1], None)
    if 'gpt' in model_name:
        full_prompt = conv_template.to_openai_api_messages()
    else:
        full_prompt = conv_template.get_prompt()
    # Clear the conv template
    # self.conv_template.messages = []
    return full_prompt

def replace_words(text, word_map):
    if not word_map:
        return text
    def replace(match):
        word = match.group(0)
        return word_map.get(word, word)
    
    # Sort keys by length descending to match longest possible word first
    sorted_keys = sorted(word_map.keys(), key=len, reverse=True)
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in sorted_keys) + r')\b')
    
    return pattern.sub(replace, text)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    print(args)
    # data_folder_path = '/root/fjl/research_works/dp_llm_work/CusText/CusText/dolly/privatized_dataset/glove.42B.300d/conservative/eps_1.0_top_20_s1_save_stop_words_True'

    with open(os.path.join(args.version_folder,args.version, 'train.json'), 'r', encoding='utf-8') as file:
            train_data = json.load(file)

    with open(os.path.join(args.version_folder,args.version, 'test.json'), 'r', encoding='utf-8') as file:
            test_data = json.load(file)

    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct' 
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


    target_model = os.environ.get("DEFAULT_MODEL", "gpt-3.5-turbo")
    llm = models.OpenAILLM(model_path=target_model)
    print(f"\n[Strict Professor Audit]: Connected to Black-box Evaluation LLM: {target_model}\n")
    
    
    # [Strict Professor Audit]: Pre-create evaluation directory to avoid runtime crashes.
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    output_filename = os.path.join(args.save_folder, args.shot + "_shot_" + ("private_" if args.use_private else "original_") + args.version.replace('/', '_') + ".json")
    
    print(f'Starting inference on {len(test_data)} samples...')

    answer_data = []
    for item in tqdm(range(len(test_data)), desc="Evaluating Utility"):
        # if item > 10:
        #     break
        
        # Decide context to use
        current_context = test_data[item]['private_context'] if args.use_private else test_data[item]['context']

        if args.shot == '2':
            # Note: For 2-shot, we should also decide whether examples should be private
            ctx1 = train_data[item*2]['private_context'] if args.use_private else train_data[item*2]['context']
            res1 = train_data[item*2]['private_response'] if args.use_private else train_data[item*2]['response']
            ctx2 = train_data[item*2+1]['private_context'] if args.use_private else train_data[item*2+1]['context']
            res2 = train_data[item*2+1]['private_response'] if args.use_private else train_data[item*2+1]['response']
            
            output = llm.generate(prompt=llm.create_conv_prompt(Summarize_Prompt_Tamplete_2_shot.format(
                dialogue1=ctx1,
                summary1=res1,
                dialogue2=ctx2,
                summary2=res2,
                private_dialogue=current_context)), 
                temperature=0.1, max_tokens=100)
        elif args.shot == '0':
            output = llm.generate(prompt=llm.create_conv_prompt(Summarize_Prompt_Tamplete_0_shot.format(
                private_dialogue=current_context)), 
                temperature=0.1, max_tokens=100)
            
        # [Strict Professor Audit]: Robust String-based Remapping (Improved from fragile token-ID matching)
        remap_sentence = output
        if args.is_map == True:
            # Build a string-level mapping dictionary
            # Warning: Some token IDs might decode with leading spaces. We need to handle that.
            str_map = {}
            for noised_id, original_id in test_data[item]['private_word_map'].items():
                noised_str = tokenizer.decode([int(noised_id)]).strip()
                original_str = tokenizer.decode([int(original_id)]).strip()
                if noised_str:
                    str_map[noised_str] = original_str
            
            # Apply substitution
            remap_sentence = replace_words(output, str_map)

        answer_data.append({
            "index": item,
            "response": test_data[item]['response'],
            "llm_response": output,
            "noised_remap_llm_response": remap_sentence,
        })
        
        # [Strict Professor Audit]: Incremental saving for robustness (Rule 4).
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(answer_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Inference completed. Results saved to: {output_filename}")

