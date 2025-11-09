# Sentiment Analysis of Product Reviews

This project implements sentiment analysis on product reviews using both Word2Vec and BERT embeddings. It compares the performance of these two approaches in classifying reviews as positive, negative, or neutral.

## Features

- Text preprocessing and cleaning
- Word2Vec embeddings with Random Forest classifier
- BERT embeddings with Logistic Regression classifier
- Model performance comparison
- Visualization of results
- Comprehensive analysis reports

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Additional setup for macOS users:
If you encounter SSL certificate issues, you may need to:
- Install certificates: Run `/Applications/Python 3.x/Install Certificates.command`
- Or set the SSL certificate path:
```bash
export SSL_CERT_FILE=/path/to/your/cacert.pem
```

## Usage

Run the main script:
```bash
python sentiment-analysis.py
```

The script will:
1. Create a sample dataset of product reviews
2. Train Word2Vec and BERT models
3. Compare model performances
4. Generate visualizations and reports

## Output

All results are saved in the `output` directory:
- Confusion matrices for both models
- Model comparison charts
- Sample predictions
- Analysis reports
- Dataset files

## Requirements

See `requirements.txt` for detailed package requirements.

## Project Structure

```
sentiment-analysis/
├── sentiment-analysis.py    # Main script
├── requirements.txt        # Package dependencies
├── output/                # Generated results
│   ├── bert_confusion_matrix.png
│   ├── word2vec_confusion_matrix.png
│   ├── model_comparison.png
│   ├── model_comparison.csv
│   ├── sample_predictions.csv
│   ├── analysis_report.json
│   └── reviews_dataset.csv
└── README.md             # This file
```