# Fake News Detection - Streamlit Web Application

A professional web application for detecting fake news using a Bidirectional LSTM model trained on the WELFake dataset.

## Features

- **Manual Input Mode**: Enter news title and body text for analysis
- **URL Scraping Mode**: Automatically extract and analyze articles from URLs
- **Real-time Predictions**: Get instant fake/real classification with confidence scores
- **Clean UI**: Modern Streamlit interface with progress bars and clear results

## Project Structure

```
Fake-News-Detection-using-Bidirectional-LSTM/
├── app/
│   ├── streamlit_app.py          # Main Streamlit application
│   └── utils/
│       ├── __init__.py           # Utils module exports
│       ├── preprocessing.py      # Text preprocessing functions
│       └── scraper.py            # URL scraping utilities
├── model/
│   └── fake_news_bilstm.h5       # Trained model file
├── notebook/
│   ├── FakeNewsDetection.ipynb   # Training notebook
│   ├── fakenewsdetection.py      # Notebook as Python script
│   └── tokenizer.pkl             # Trained tokenizer (generate from notebook)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Prerequisites

Before running the application, ensure you have:

1. **Python 3.8+** installed
2. **Trained model file**: `model/fake_news_bilstm.h5`
3. **Tokenizer file**: `notebook/tokenizer.pkl`

> **Note**: The tokenizer file is generated when you run the training notebook. Make sure to run the "Save Model" cell in the notebook to create it.

## Installation

1. Clone the repository or navigate to the project directory:
   ```bash
   cd Fake-News-Detection-using-Bidirectional-LSTM
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage

### Manual Input Mode

1. Select "Manual Input" mode
2. Enter the news article **title**
3. Enter the news article **body text**
4. Click "Analyze"

The system will:
- Combine the title with the first meaningful paragraph
- Process through the trained model
- Display prediction results with confidence scores

### URL Scraping Mode

1. Select "URL Scraping" mode
2. Paste a news article URL
3. Click "Scrape & Analyze"

The system will:
- Extract the article title and content
- Filter out navigation text, ads, and footers
- Use only the first meaningful paragraph
- Display extracted content for verification
- Run prediction and show results

## Model Details

| Parameter | Value |
|-----------|-------|
| Architecture | Bidirectional LSTM |
| Embedding Size | 128 |
| LSTM Units | 128 |
| Max Sequence Length | 500 |
| Vocabulary Size | 20,000 |
| Output | Binary (Real=1, Fake=0) |

## Important Notes

### Input Format Requirements

For best results, the input should match the training distribution:
- **Title + First paragraph** of the article body
- Clean text without navigation, ads, or footer content
- Minimum 50 words recommended

### Limitations

1. **Stylistic Detection**: The model detects linguistic patterns, not factual accuracy
2. **Distribution Sensitivity**: Performance may vary for content outside training distribution
3. **Language**: Optimized for English content
4. **Short Text**: Very short inputs may produce unreliable predictions

### Disclaimer

> This model detects stylistic patterns based on the training dataset (WELFake). 
> It does **not** verify factual truth. Always cross-reference with trusted sources.

## Troubleshooting

### Model Loading Error
- Ensure `model/fake_news_bilstm.h5` exists
- Check TensorFlow version compatibility

### Tokenizer Loading Error
- Run the training notebook to generate `tokenizer.pkl`
- Ensure the file is in `notebook/tokenizer.pkl`

### Scraping Errors
- Some websites block automated requests
- Check your internet connection
- Try a different news source

## License

This project is for educational purposes.

## Acknowledgments

- Dataset: [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- Framework: TensorFlow/Keras, Streamlit
- Scraping: newspaper3k
