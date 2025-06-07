# BLC (Binary Language Classifier)

## Introduction

BLC (Binary Language Classifier) is a project to detect whether an input text is in the Italian language or not. It uses a deep learning model trained on a public dataset from [Kaggle](https://www.kaggle.com/datasets/basilb2s/language-detection) to perform binary classification. The solution exposes a local REST API endpoint for inference, allowing the user to send a text and receive a prediction (Italian or non-Italian).

## Repository Structure

- `dataset/` : Contains the dataset used for training.
    - `dataset_info.md` : Information about the dataset.
    - `language_detection.csv` : Dataset downloaded from Kaggle (see below).
- `model/` : Contains the trained model and the vocabulary for text vectorization.
    - `blc_classifier_weights.pth` : Weights of the trained classifier.
    - `blc_config.json` : Some config parameters for the model and for the vocabulary.
    - `blc_vocab.json` : Vocabulary data for text vectorization.
    - `model_info.md` : Information about the model. (TODO ------)
- `src/` : Contains the source code for the task.
    - `client.py` : Interactive demo client for testing the API from the terminal.
    - `server.py` : Trains the model (if needed) and starts the Flask API server.
- `requirements.txt` : Contains the Python dependencies required for the task.


## Usage

### 0. Prerequisites:
- Python 3.9+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/NicoloResmini/BLC.git
```

### 2. Create a virtual environment and install dependencies

```bash
cd BLC
python -m venv my_venv
source my_venv/bin/activate  # On Windows (CMD): my_venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the server

```bash
cd src
python server.py
```

This will train the model (if not already trained) and start the API server at `http://localhost:5000`.

### 4. Run the client

In a separate terminal window, activate the virtual environment again and run the client:

```bash
cd BLC
source my_venv/bin/activate  # On Windows (CMD): my_venv\Scripts\activate
cd src
python client.py
```

This will start an interactive client that allows you to enter text and get predictions via API.