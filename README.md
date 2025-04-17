# Legal Text Summarization System

A deep learning-based system for summarizing Indian legal documents, combining extractive summarization with legal context analysis.

## Features

- Extractive summarization of legal documents using deep learning
- IPC (Indian Penal Code) section detection and analysis
- Comparative analysis of traditional vs. deep learning approaches
- High-quality legal text processing and analysis
- ROUGE score evaluation and performance metrics

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd legal-text-summarization
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
legal-text-summarization/
├── data/
│   ├── final_data.csv
│   ├── extractive_summaries.csv
│   └── ipc_sections.csv
├── models/
│   ├── extractive_summarizer_model.pth
│   └── saved_tokenizer/
├── notebooks/
│   ├── Create Dataset.ipynb
│   └── Extractive Generation.ipynb
├── src/
│   └── model.py
├── requirements.txt
└── README.md
```

## Usage

### 1. Dataset Creation

The project uses the Indian Legal Text dataset. To create the dataset:

```python
from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset = load_dataset("Yashaswat/Indian-Legal-Text-ABS")
split = dataset['train'].select(range(2000))

# Process and save
df = pd.DataFrame(split)
df.to_csv("final_data.csv", index=False)
```

### 2. Model Training

To train the extractive summarization model:

```python
from src.model import ExtractiveSummarizer
import torch

# Initialize model
model = ExtractiveSummarizer(
    vocab_size=30522,
    embedding_dim=256,
    hidden_dim=128,
    output_dim=1,
    n_layers=2,
    dropout=0.3
)

# Train model
# See notebooks/Extractive Generation.ipynb for detailed training process
```

### 3. Generating Summaries

To generate summaries for legal documents:

```python
from transformers import BertTokenizer
import torch

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained("saved_tokenizer/")
model = ExtractiveSummarizer()
model.load_state_dict(torch.load("models/extractive_summarizer_model.pth"))

# Generate summary
def summarize_text(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?])\\s+', text.strip())
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        scores = model(inputs["input_ids"])
    
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_sentences]
    summary = " ".join([sentences[i] for i in sorted(top_idxs)])
    return summary
```

### 4. IPC Section Analysis

To analyze IPC sections in legal documents:

```python
import re
import pandas as pd

def extract_ipc_sections(text):
    matches = re.findall(r"Section\\s+(\\d+[A-Z]?)", text, flags=re.IGNORECASE)
    return list(set([f"IPC_{match.upper()}" for match in matches]))

# Load IPC sections data
ipc_df = pd.read_csv("data/ipc_sections.csv")
```

## Performance Metrics

The model achieves the following ROUGE scores:
- ROUGE-1: 60.10%
- ROUGE-2: 41.85%
- ROUGE-L: 32.66%

## Comparative Analysis

Traditional methods performance:
- TF-IDF: 48.92% (ROUGE-1)
- LSA: 53.48% (ROUGE-1)
- TextRank: 53.68% (ROUGE-1)

## Requirements

```
datasets==2.10.0
rouge_score==0.1.2
tabulate==0.9.0
tqdm==4.67.1
torch>=1.8.0
transformers>=4.0.0
nltk>=3.6.0
pandas>=1.2.0
numpy>=1.19.0
scikit-learn>=0.24.0
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Indian Legal Text dataset from Hugging Face
- LED model from AllenAI
- BERT tokenizer and model from Google Research
