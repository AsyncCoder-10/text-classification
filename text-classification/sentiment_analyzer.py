import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)


def getsentiment(text: str) -> str:
    """Return the sentiment label for a piece of text."""
    if not isinstance(text, str) or not text.strip():
        return "N/A"

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]


def analyze_reviews(file_path: str, review_column: str = "Reviews") -> pd.DataFrame:
    """
    Read an Excel file of reviews and return a DataFrame with an
    added 'Sentiment' column.
    """
    df = pd.read_excel(file_path)

    if review_column not in df.columns:
        raise ValueError(
            f"Column '{review_column}' not found in file. Available columns: {list(df.columns)}"
        )

    df["Sentiment"] = df[review_column].apply(getsentiment)
    return df


def process_file(file):
    if file is None:
        return pd.DataFrame({"Error": ["Please upload an Excel file."]})

    try:
        return analyze_reviews(file.name)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})


demo = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="Upload Excel File", file_types=[".xlsx"]),
    outputs=gr.Dataframe(label="Reviews with Sentiment"),
    title="Review Sentiment Analyzer",
    description="Upload an Excel file with a 'Reviews' column to get sentiment for each review.",
)

if __name__ == "__main__":
    demo.launch()
