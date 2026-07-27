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


def analyze_sentiment(text: str) -> str:
    if not text.strip():
        return "Please enter some text."

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    label = model.config.id2label[predicted_class_id]
    score = torch.softmax(logits, dim=-1)[0][predicted_class_id].item()

    return f"{label} ({score:.2%} confidence)"


demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter text here...", label="Input Text"),
    outputs=gr.Textbox(label="Sentiment"),
    title="Sentiment Analyzer",
    description="Classify text sentiment using DistilBERT (SST-2).",
    examples=[["This product is good"], ["This product was quite expensive"]],
)

if __name__ == "__main__":
    demo.launch()
