# Binary Language Classifier

## Introduction

BLC (Binary Language Classifier) is a machine learning model that performs binary classification to determine whether an input text is written in Italian or not. Trained on a public Kaggle [dataset](https://www.kaggle.com/datasets/basilb2s/language-detection) for language detection, the model serves predictions through a local REST API endpoint, classifying the submitted text in "Italian" or "non-Italian".


## Usage

### 0. Prerequisites:
- Python 3.9+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/NicoloResmini/binary-language-classifier.git
```

### 2. Create a virtual environment and install dependencies

```bash
cd binary-language-classifier
python -m venv my_venv
source my_venv/bin/activate  # On Windows (CMD): my_venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the server

```bash
cd src
python server.py
```

This will train the model (if not already stored in 'model') and start the API server at `http://localhost:5000`.

### 4. Run the client

In a separate terminal window, activate the virtual environment and then run the client:

```bash
cd binary-language-classifier
source my_venv/bin/activate  # On Windows (CMD): my_venv\Scripts\activate
cd src
python client.py
```

This will start an interactive session where the user can enter a text and get the prediction via API.
