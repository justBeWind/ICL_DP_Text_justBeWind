import json
import argparse
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to the evaluated JSON file')
    return parser

def evaluate_predictions(references, predictions, metric_name=""):
    print(f"\n========== Evaluating {metric_name} ==========")
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}
    
    print("Computing ROUGE...")
    for ref, pred in tqdm(zip(references, predictions), total=len(references), desc="ROUGE"):
        scores = scorer.score(ref, ref if pred is None else pred) # handle None gracefully
        for key in rouge_scores.keys():
            rouge_scores[key].append(scores[key].fmeasure)
            
    avg_rouge = {k: np.mean(v) for k, v in rouge_scores.items()}
    
    print("\n--- ROUGE Scores ---")
    for k, v in avg_rouge.items():
        print(f"{k}: {v:.4f}")
        
    # BERTScore
    print("\nComputing BERTScore...")
    # Using 'en' default model (usually roberta-large). Replace None with empty strings if any
    clean_preds = [p if p is not None else "" for p in predictions]
    P, R, F1 = score(clean_preds, references, lang='en', verbose=True)
    
    print("\n--- BERTScore ---")
    print(f"Precision: {P.mean().item():.4f}")
    print(f"Recall: {R.mean().item():.4f}")
    print(f"F1: {F1.mean().item():.4f}")

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    references = []
    preds_nomap = []
    preds_map = []
    
    for item in data:
        references.append(item.get('response', ""))
        preds_nomap.append(item.get('llm_response', ""))
        preds_map.append(item.get('noised_remap_llm_response', ""))
        
    print(f"Total samples: {len(data)}")
    
    evaluate_predictions(references, preds_nomap, "Ours (no map)")
    evaluate_predictions(references, preds_map, "Ours (map)")

if __name__ == '__main__':
    main()
