# 📄 AI Legal Sentiment Analyzer

This project analyzes legal documents and classifies their **sentiment** as **POSITIVE**, **NEGATIVE**, or **NEUTRAL** using a pre-trained Transformer model. It's built using **Google Colab** and **Hugging Face Transformers**, with no external API or cloud service required.

---

## 🚀 Features

- Sentiment analysis tailored to legal language  
- Classifies legal text into **POSITIVE**, **NEGATIVE**, or **NEUTRAL**  
- Simple CSV input/output for legal documents  
- Fully beginner-friendly  
- Uses a pre-trained BERT model via Hugging Face

---

## 🛠️ Tech Stack

- [Google Colab](https://colab.research.google.com)
- [Transformers by Hugging Face](https://huggingface.co/transformers/)
- Python (Pandas)

---

## 📚 How It Works

### 🔹 Load the pre-trained sentiment analysis model

```python
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
```

---

### 🔹 Run sentiment analysis on legal text data

```python
import pandas as pd

df = pd.read_csv("legal_docs.csv")
results = [sentiment_pipeline(text)[0]["label"] for text in df["text"]]
df["stars"] = results
```

---

### 🔹 Map stars to sentiment labels

```python
def map_sentiment(star_label):
    stars = int(star_label[0])
    if stars <= 2:
        return "NEGATIVE"
    elif stars == 3:
        return "NEUTRAL"
    else:
        return "POSITIVE"

df["sentiment"] = df["stars"].apply(map_sentiment)
df.to_csv("labeled_outputs.csv", index=False)
```

---

### 🔹 Download the result

```python
from google.colab import files
files.download("labeled_outputs.csv")
```

---

## 🧪 Sample Input Format (`legal_docs.csv`)

```csv
id,text
1,"The agreement benefits both parties involved."
2,"The defendant must pay $10,000 in damages."
3,"This contract outlines responsibilities."
```

---

## ✅ Output Format (`labeled_outputs.csv`)

```csv
id,text,stars,sentiment
1,"The agreement benefits both parties involved.","5 stars",POSITIVE
2,"The defendant must pay $10,000 in damages.","2 stars",NEGATIVE
3,"This contract outlines responsibilities.","3 stars",NEUTRAL
```

---

## 📌 Future Improvements

- 🔧 Train or fine-tune a legal-specific sentiment model  
- 🏷️ Add multi-label tagging (e.g., "favorable to plaintiff", "penalty for defendant")  
- 💻 Build a Web UI using Streamlit or Gradio  

---


## 📃 License

This project is open-source and available under the [MIT License](LICENSE).
