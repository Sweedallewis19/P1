import pytesseract
from PIL import Image, ImageDraw, ImageFont
import re
import torch
import glob
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import textwrap
import os

# Load summarization model (BART)
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Clean raw OCR text
def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r"[^a-zA-Z0-9.,%+\-/()\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Split text into sections like 1. CHF, 2. HTN etc.
def split_sections(text):
    pattern = r"(?=\d\.\s)"
    sections = re.split(pattern, text)
    sections = [clean_text(sec) for sec in sections if sec.strip()]
    if len(sections) <= 1:
        return [clean_text(text)]
    return sections

# Summarize using BART
def summarize_section(section_text):
    input_text = "summarize: " + section_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(
        **inputs,
        max_length=180,
        num_beams=4,
        repetition_penalty=1.2,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Get all image paths
image_paths = glob.glob("images/*.png") + glob.glob("images/*.jpg") + glob.glob("images/*.jpeg")
if not image_paths:
    print("No images found in 'images/' folder.")
    exit()

# Store summaries
full_summary_report = []

for img_index, img_path in enumerate(image_paths, 1):
    print(f"\nProcessing Image {img_index}: {img_path}")
    try:
        img = Image.open(img_path)
        ocr_text = pytesseract.image_to_string(img)

        if not ocr_text.strip():
            print("No text found in the image.")
            continue

        sections = split_sections(ocr_text)
        print(f"🔍 Found {len(sections)} section(s).\n")

        img_summary = [f"Summary for Image {img_index}: {img_path}"]

        for i, sec in enumerate(sections, 1):
            try:
                summary = summarize_section(sec)
                img_summary.append(f"🔹 Section {i}: {summary}")
            except Exception as e:
                img_summary.append(f" Failed to summarize Section {i}: {e}")

        full_summary_report.append("\n".join(img_summary))

    except Exception as e:
        full_summary_report.append(f" Error processing {img_path}: {e}")

# Combine into one big string
combined_summary_text = "\n\n".join(full_summary_report)
print("\n\n Final Combined Summary Report:\n")
print(combined_summary_text)

# ✨ Create a single image of all summaries
def create_summary_image(text, output_path="summary_output.png", font_size=18, max_width=1000):
    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    lines = []
    wrapper = textwrap.TextWrapper(width=100)
    for para in text.split("\n"):
        lines.extend(wrapper.wrap(para))
        lines.append("")  # empty line between paragraphs

    # Calculate image size
    line_height = font.getbbox('A')[3] + 10
    img_height = line_height * len(lines) + 40

    img = Image.new("RGB", (max_width, img_height), color="white")
    draw = ImageDraw.Draw(img)

    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black", font=font)
        y += line_height

    img.save(output_path)
    print(f"\nSummary image saved as '{output_path}'")

# Generate the image
create_summary_image(combined_summary_text)
