import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def model_fn(model_dir):
    """Load the model for inference.
    Args:
        model_dir (str): Path to the model directory.
    Returns:
        dict: A dictionary containing the loaded model and tokenizer.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, problem_type="multi_label_classification")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return {"model": model, "tokenizer": tokenizer}

def predict_fn(data, model_dict):
    """Perform prediction on the input data using the loaded model and tokenizer.
    Args:
        data (dict): Input data containing the text to classify.
        model_dict (dict): Dictionary containing the loaded model and tokenizer.

    Returns:
        list: List of prediction results with labels and scores.
    """

    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]

    # preprocess the input text
    text_input = data["inputs"]
    text_input = " ".join(text_input.split())

    inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        logits = torch.sigmoid(torch.tensor(logits)).cpu().detach().numpy()

        # define different thresholds for the first four labels
        thresholds = np.ones_like(logits)  
        thresholds[:, :4] = [0.2, 0.5, 0.8, 0.95]  

        # sort logits from highest to lowest
        sorted_indices = np.argsort(logits, axis=1)[:, ::-1]  
        reverse_indices = np.argsort(sorted_indices, axis=1)  

        sorted_logits = np.take_along_axis(logits, sorted_indices, axis=1)  
        sorted_predictions = np.where(sorted_logits >= thresholds, 1, 0)  

        # rearrange predictions back to original order
        predictions = np.take_along_axis(sorted_predictions, reverse_indices, axis=1) 
        
        # now only keep predictions and logits for the predictions that are 1
        results = []

        for i in range(predictions.shape[0]):
            pred_indices = np.where(predictions[i] == 1)[0]
            pred_names = [model.config.id2label[idx] for idx in pred_indices]
            pred_probs = [logits[i][idx] for idx in pred_indices]
            results.append({
                "label": pred_names,
                "score": pred_probs
            })

    return results

