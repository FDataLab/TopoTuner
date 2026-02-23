import re
import numpy as np
from rouge_score import rouge_scorer


def parse_tagged_spans(text, tag_name):
    """
    Extracts the first occurrence of text within a given XML-like tag.
    Example: parse_tagged_spans("<cause>This is it</cause>", "cause") -> "This is it"
    Returns an empty string if the tag is not found or is empty.
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, text, re.DOTALL) # re.DOTALL allows '.' to match newlines
    if matches:
        return matches[0].strip() # Return the content of the first found tag
    return "" # Return empty if tag not found

def compute_metrics_for_causal_extraction(eval_preds, tokenizer_object):
    """
    Computes Precision, Recall, F1, and Exact Match for cause/effect extraction.
    eval_preds: A tuple (predictions, label_ids)
    tokenizer_object: The tokenizer used for encoding/decoding.
    """
    predictions_logits, label_ids = eval_preds

    # Decode predictions (convert logits to token IDs, then token IDs to text)
    # For CausalLM, predictions are usually logits.
    predicted_token_ids = np.argmax(predictions_logits, axis=-1)
    decoded_predictions = tokenizer_object.batch_decode(predicted_token_ids, skip_special_tokens=True)

    # Decode label_ids (ground truth). Replace -100s used for masking.
    # -100 is often used for tokens that should not contribute to the loss (e.g., prompt tokens)
    label_ids_cleaned = np.where(label_ids != -100, label_ids, tokenizer_object.pad_token_id)
    decoded_labels = tokenizer_object.batch_decode(label_ids_cleaned, skip_special_tokens=True)

    true_causes_list = []
    pred_causes_list = []
    true_effects_list = []
    pred_effects_list = []
    
    # rouge score calculation
    rouge_types = ['rougeL'] 
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True) 

    all_cause_rougeL_fmeasure = []
    all_effect_rougeL_fmeasure = []
    
    # calculate precision, recall, f1, exact match for cause and effect
    for i in range(len(decoded_labels)):
        true_full_text = decoded_labels[i]
        pred_full_text = decoded_predictions[i]

        true_c = parse_tagged_spans(true_full_text, "cause")
        true_e = parse_tagged_spans(true_full_text, "effect")
        true_causes_list.append(true_c)
        true_effects_list.append(true_e)

        pred_c = parse_tagged_spans(pred_full_text, "cause")
        pred_e = parse_tagged_spans(pred_full_text, "effect")
        pred_causes_list.append(pred_c)
        pred_effects_list.append(pred_e)

        # --- ADDED: Calculate ROUGE-L for non-empty true spans ---
        if true_c: # Only score if ground truth cause is not empty
            scores_c = scorer.score(target=true_c, prediction=pred_c)
            all_cause_rougeL_fmeasure.append(scores_c['rougeL'].fmeasure)
        
        if true_e: # Only score if ground truth effect is not empty
            scores_e = scorer.score(target=true_e, prediction=pred_e)
            all_effect_rougeL_fmeasure.append(scores_e['rougeL'].fmeasure)

    metrics = {}

    # Calculate metrics for "cause"
    tp_cause = sum(1 for tc, pc in zip(true_causes_list, pred_causes_list) if tc != "" and tc == pc)
    fp_cause = sum(1 for tc, pc in zip(true_causes_list, pred_causes_list) if pc != "" and tc != pc) # Predicted but wrong, or predicted when shouldn't have
    fn_cause = sum(1 for tc, pc in zip(true_causes_list, pred_causes_list) if tc != "" and tc != pc) # Should have predicted but didn't, or predicted wrong

    precision_cause = tp_cause / (tp_cause + fp_cause) if (tp_cause + fp_cause) > 0 else 0.0
    recall_cause = tp_cause / (tp_cause + fn_cause) if (tp_cause + fn_cause) > 0 else 0.0
    f1_cause = 2 * (precision_cause * recall_cause) / (precision_cause + recall_cause) if (precision_cause + recall_cause) > 0 else 0.0
    
    metrics["cause_precision"] = precision_cause
    metrics["cause_recall"] = recall_cause
    metrics["cause_f1"] = f1_cause
    
    # Exact Match for non-empty causes
    total_true_causes_present = sum(1 for tc in true_causes_list if tc != "")
    correct_cause_matches = sum(1 for tc, pc in zip(true_causes_list, pred_causes_list) if tc != "" and tc == pc)
    metrics["cause_exact_match_accuracy"] = correct_cause_matches / total_true_causes_present if total_true_causes_present > 0 else 0.0


    # Calculate metrics for "effect"
    tp_effect = sum(1 for te, pe in zip(true_effects_list, pred_effects_list) if te != "" and te == pe)
    fp_effect = sum(1 for te, pe in zip(true_effects_list, pred_effects_list) if pe != "" and te != pe)
    fn_effect = sum(1 for te, pe in zip(true_effects_list, pred_effects_list) if te != "" and te != pe)

    precision_effect = tp_effect / (tp_effect + fp_effect) if (tp_effect + fp_effect) > 0 else 0.0
    recall_effect = tp_effect / (tp_effect + fn_effect) if (tp_effect + fn_effect) > 0 else 0.0
    f1_effect = 2 * (precision_effect * recall_effect) / (precision_effect + recall_effect) if (precision_effect + recall_effect) > 0 else 0.0

    metrics["effect_precision"] = precision_effect
    metrics["effect_recall"] = recall_effect
    metrics["effect_f1"] = f1_effect

    total_true_effects_present = sum(1 for te in true_effects_list if te != "")
    correct_effect_matches = sum(1 for te, pe in zip(true_effects_list, pred_effects_list) if te != "" and te == pe)
    metrics["effect_exact_match_accuracy"] = correct_effect_matches / total_true_effects_present if total_true_effects_present > 0 else 0.0

    # Instance-level accuracy: proportion of examples where BOTH cause and effect are perfectly handled
    # (i.e., if true_X is present, pred_X matches it; if true_X is empty, pred_X is also empty)
    perfect_instances = 0
    for i in range(len(decoded_labels)):
        cause_match = (true_causes_list[i] == pred_causes_list[i])
        effect_match = (true_effects_list[i] == pred_effects_list[i])
        if cause_match and effect_match:
            perfect_instances += 1
    
    metrics["instance_perfect_match_accuracy"] = perfect_instances / len(decoded_labels) if len(decoded_labels) > 0 else 0.0
    
    metrics["cause_rougeL_fmeasure"] = np.mean(all_cause_rougeL_fmeasure) if all_cause_rougeL_fmeasure else 0.0
    metrics["effect_rougeL_fmeasure"] = np.mean(all_effect_rougeL_fmeasure) if all_effect_rougeL_fmeasure else 0.0
    
    
    return metrics