# BLC

## Introduction

BLC (Binary Language Classifier) is a project to detect whether an input text is in the Italian language or not. It uses a machine learning model trained on a public dataset from [Kaggle](https://www.kaggle.com/datasets/basilb2s/language-detection) to perform binary classification. The solution exposes a local REST API endpoint for inference, allowing you to send a text and receive a prediction (Italian or non-Italian).

## Repository Structure

- `main.py` : Trains the model (if needed) and starts the Flask API server.
- `demo.py` : Interactive demo client for testing the API from the terminal.
- `dataset.csv` : Dataset for training (downloaded from Kaggle, see below).
- `model.pkl` : Saved trained model (auto-generated after training).
- `requirements.txt` : Python dependencies.
- `instructions.pdf` : Project instructions.

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/NicoloResmini/BLC
cd BLC
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the server

```bash
python main.py
```

This will train the model (if not already trained) and start the API server at `http://localhost:5000`.

### 4. Run the client

In a separate terminal:

```bash
python demo.py
```

This will start an interactive client that allows you to enter text and get predictions from the API.