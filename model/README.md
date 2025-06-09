# Model Information

## Model Architecture

The model is a feedforward neural network designed for binary classification of Italian vs non-Italian texts:

```
Input Layer:    10,000 features (n-gram embeddings)
Hidden Layer:   Linear(10000 → 128) + ReLU
Output Layer:   Linear(128 → 1) + Sigmoid
```

## Feature Engineering

### Text Preprocessing

Before feature extraction, input text undergoes several preprocessing steps:

1. **Lowercase Conversion**: All text is converted to lowercase for case-insensitive analysis
2. **Punctuation Removal**: Standard ASCII punctuation marks are removed while preserving Unicode characters (e.g., accented letters like à, è, ò)
3. **Number Removal**: All numeric digits are stripped from the text
4. **Whitespace Normalization**: Multiple consecutive spaces are collapsed into single spaces
5. **Trimming**: Leading and trailing whitespace is removed

This preprocessing ensures consistent text representation while maintaining language-specific characters essential for Italian text classification.

### Text Embedding
The model uses **character n-grams** to represent input text:

- **N-gram Range**: 2-4 characters (bi-gram, tri-gram, 4-gram)
- **Vocabulary Size**: 10,000 most frequent n-grams from training set
- **Representation**: Sparse binary vectors (bag-of-n-grams)

The preprocessed text is thus converted into numerical features through the following process: 
1. Extract all possible n-grams of length 2, 3, and 4 characters from input text
2. Create a binary vector of 10,000 dimensions
3. For each of the 10,000 n-grams in the vocabulary, set the value to 1 if it is present in the input text, 0 otherwise
4. N-grams are ordered by decreasing frequency in training set

## Training Configuration

- **Train/Test Split**: 80% / 20%
- **Loss Function**: Binary Cross Entropy (BCE)
- **Optimizer**: Adam
- **Learning Rate**: 1e-3 (0.001)
- **Batch Size**: 32
- **Epochs**: 10

## Model Files

- **`blc_classifier_weights.pth`**: PyTorch model weights for loading pre-trained model
- **`blc_config.json`**: Configuration settings including n-gram range, vocabulary size, and network dimensions
- **`blc_vocab.json`**: Vocabulary mapping of 10,000 most frequent n-grams to their indices, ordered by decreasing frequency in training set