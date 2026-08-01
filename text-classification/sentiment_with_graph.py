import pandas as pd
import torch
import matplotlib.pyplot as plt
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


def get_sentiment_bar_chart(df: pd.DataFrame, sentiment_column: str = "Sentiment"):
    """
    Build a bar chart showing the count of each sentiment category.

    Returns:
        A matplotlib.figure.Figure object showing sentiment counts.
    """
    if sentiment_column not in df.columns:
        raise ValueError(f"Column '{sentiment_column}' not found in DataFrame.")

    sentiment_counts = df[sentiment_column].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        sentiment_counts.index, sentiment_counts.values, color=["#4CAF50", "#F44336"]
    )

    ax.set_title("Sentiment Distribution")
    ax.set_xlabel(sentiment_column)
    ax.set_ylabel("Count")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            str(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
        )

    fig.tight_layout()
    return fig


def process_file(file):
    if file is None:
        return pd.DataFrame({"Error": ["Please upload an Excel file."]}), None

    try:
        df = analyze_reviews(file.name)
        chart = get_sentiment_bar_chart(df)
        return df, chart
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]}), None


with gr.Blocks(title="Review Sentiment Analyzer") as demo:
    gr.Markdown("## Review Sentiment Analyzer")
    gr.Markdown(
        "Upload an Excel file with a 'Reviews' column to get sentiment for each review."
    )

    file_input = gr.File(label="Upload Excel File", file_types=[".xlsx"])
    submit_btn = gr.Button("Analyze")

    output_table = gr.Dataframe(label="Reviews with Sentiment")
    output_chart = gr.Plot(label="Sentiment Distribution")

    submit_btn.click(
        fn=process_file, inputs=file_input, outputs=[output_table, output_chart]
    )

if __name__ == "__main__":
    demo.launch()
