import os
import re
import nltk
import torch
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn import functional as F
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Load FinBERT
print("Loading FinBERT model...")
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
model.eval()

# Get latest filing from directory
filing_dir = "filings"
filing_files = [f for f in os.listdir(filing_dir) if f.endswith(".txt")]
if not filing_files:
    print("No filings found.")
    exit()
latest_file = sorted(filing_files)[-1]
print(f"Analyzing: {latest_file}")
with open(os.path.join(filing_dir, latest_file), "r", encoding="utf-8") as file:
    raw_text = file.read()

# Extract sentences related to risk
sentences = sent_tokenize(raw_text)
risk_sentences = [s for s in sentences if "risk" in s.lower()]

print(f"Found {len(risk_sentences)} risk-related sentences.")

# Clean the sentences
stop_words = set(stopwords.words("english"))
filtered = []
for s in risk_sentences:
    words = re.findall(r'\b\w+\b', s.lower())
    if any(w not in stop_words for w in words):
        filtered.append(s)

def get_finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probs, dim=-1).item()
        labels = ['positive', 'neutral', 'negative']
        return labels[sentiment], round(max(probs.tolist()[0]), 3)

# Run sentiment analysis with progress bar
data = []
for sent in tqdm(filtered, desc="Analyzing sentences"):
    try:
        label, conf = get_finbert_sentiment(sent)
        data.append({
            "sentence": sent.strip(),
            "sentiment": label,
            "confidence": conf
        })
    except Exception as e:
        print(f"⚠️ Skipped a sentence due to error: {e}")

# Save results
if data:
    df = pd.DataFrame(data)
    df.to_csv("output/finbert_results.csv", index=False)
    print("✅ Sentiment analysis complete. Results saved to output/finbert_results.csv")
else:
    print("❌ No usable sentiment results.")
