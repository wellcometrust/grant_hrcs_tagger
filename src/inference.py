import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def model_fn(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, problem_type="multi_label_classification")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return {"model": model, "tokenizer": tokenizer}

def predict_fn(data, model_dict):
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    inputs = tokenizer(data["inputs"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        logits = torch.sigmoid(torch.tensor(logits)).cpu().detach().numpy()

        thresholds = [1] * logits.shape[1]
        thresholds[:4] = [0.2, 0.5, 0.8, 0.95]

        # Prepare an array to hold your predictions
        predictions = np.zeros_like(logits)

        # Loop through each sample's logits
        for i, logit in enumerate(logits):
            # Get the indices of the logits sorted by value in descending order
            sorted_indices = np.argsort(logit)[::-1]

            # Assign 1 to the top logits that exceed their respective thresholds
            for rank, idx in enumerate(sorted_indices):
                if logit[idx] > thresholds[rank]:
                    predictions[i, idx] = 1
        



        # now only keep predictions and logits for the predictions that are 1
        predictions_name = []
        predictions_probabilities = []
        for i in range(predictions.shape[0]):
            pred_indices = np.where(predictions[i] == 1)[0]
            pred_names = [model.config.id2label[idx] for idx in pred_indices]
            pred_probs = [logits[i][idx] for idx in pred_indices]
            predictions_name.append(pred_names)
            predictions_probabilities.append(pred_probs)

        results = []
        for names, probs in zip(predictions_name, predictions_probabilities):
            results.append({
                "label": names,
                "score": probs
            })

    return results

