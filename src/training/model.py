from transformers import (
    AutoModelForSequenceClassification,
    DistilBertForSequenceClassification,
    ModernBertForSequenceClassification,
)


def get_model(model_name, num_labels, label_names):
    """
    Loads a pre-trained model for sequence classification.

    Args:
        model_name (str): The name of the model to load.
        num_labels (int): The number of labels for classification.
        label_names (dict): A dictionary mapping label IDs to label names.

    Returns:
        transformers.PreTrainedModel: The loaded model.
    """
    model_class = AutoModelForSequenceClassification
    if "modernbert" in model_name.lower():
        model_class = ModernBertForSequenceClassification
    elif "distilbert" in model_name.lower():
        model_class = DistilBertForSequenceClassification

    model = model_class.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        id2label=label_names,
    )
    print(f"Model initialized using {model_class.__name__}")

    return model
