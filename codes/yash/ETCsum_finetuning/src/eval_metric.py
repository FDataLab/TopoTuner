import numpy as np
import torch
import evaluate
from bert_score import score

rouge_metric = evaluate.load("rouge")
def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred

    # Predictions are logits, so we need to take the argmax to get token IDs
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Some models return multiple outputs

    if isinstance(predictions, torch.Tensor):
        pred_tokens = torch.argmax(predictions, dim=-1)
    else:
        pred_tokens = np.argmax(predictions, axis=-1)

    decoded_preds = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)

    # Replace -100 in the labels as we don't want to decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a list of predictions and a list of references
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    print("Rouge Result:", rouge_result)

    # Extract a few key Rouge metrics
    # rouge_result = {f"rouge_{key}": value.mid.fmeasure * 100 for key, value in rouge_result.items()}

    # Calculate BERTScore
    P, R, F1 = score(decoded_preds, decoded_labels, lang='en', verbose=False)
    bertscore_result = {
        "bertscore_precision": P.mean().item() * 100,
        "bertscore_recall": R.mean().item() * 100,
        "bertscore_f1": F1.mean().item() * 100,
    }

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in pred_tokens]
    gen_len = np.mean(prediction_lens)
    result = {**rouge_result, **bertscore_result, "gen_len": round(gen_len, 4)}
    result = {k: round(v, 4) for k, v in result.items()}
    return result