import pytesseract
from PIL import Image
import pdfplumber
from transformers import pipeline
import os
import re

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def clean_text(text):
    # Remove whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def summarize_text(text):
    if len(text) < 50:
        return "Text too short to summarize."

    max_input_len = min(len(text), 2048)  # BART input limit
    summary = summarizer(
        text[:max_input_len],
        max_length=300,
        min_length=100,
        do_sample=False
    )
    return summary[0]['summary_text']

def summarize_news(file_path):
    if not os.path.exists(file_path):
        print(" File not found.")
        return

    if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        text = extract_text_from_image(file_path)
    elif file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    else:
        print(" Unsupported file format. Use a PDF or an image.")
        return

    text = clean_text(text)

    


    print("\n Generating Summary...\n")
    summary = summarize_text(text)
    print(" Summary:\n")
    print(summary)

if __name__ == "__main__":
    file_path = input("Enter path to news image or PDF: ").strip()
    summarize_news(file_path)
