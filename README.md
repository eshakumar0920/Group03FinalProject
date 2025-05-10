
# Generational Slang Classification using BERT

This project explores how slang can be used to identify generational language patterns — focusing on Boomers, Gen X, Millennials, and Gen Z. We combine rule-based techniques with modern NLP tools like BERT and machine learning models to test classification accuracy across both clean and noisy data.

---

## Table of Contents

- [Slang Analyzer](#slang-analyzer)  
- [Improved Classification](#improved-classification)  
- [Comparison Over Generations](#comparison-over-generations)  
- [Dataset Implementation](#dataset-implementation)  
- [Feedforward Neural Network (FFNN)](#feedforward-neural-network-ffnn)  
- [Logistic Regression and SVM](#logistic-regression-and-svm)  
- [Key Takeaways](#key-takeaways)  
- [Future Work](#future-work)  
- [How to Run](#how-to-run)  
- [Requirements](#requirements)  
- [Project Structure](#project-structure)  

---

## Slang Analyzer

A rule-based scorer was built using generational slang dictionaries. The analyzer scans sentences for matching terms and tallies scores based on frequency and relevance. This fast, interpretable approach performs well on clean datasets but lacks flexibility on ambiguous inputs.

---

## Improved Classification

To move beyond simple dictionary matching, we used BERT embeddings for richer sentence representations. These embeddings, which capture context and semantics, were used as input features for more advanced classifiers like FFNN, Logistic Regression, and SVM. This improved classification, particularly for clean inputs.

---

## Comparison Over Generations

This section evaluates model performance by generation. It highlights challenges in distinguishing generations with overlapping slang and showcases where models struggled, especially on noisy, mixed-generation sentences.

---

## Dataset Implementation

We created a synthetic dataset of 20,000 sentences with the following characteristics:

- Slang terms pulled from curated generational dictionaries  
- Sentences labeled by generation  
- Noise added via typos and neutral phrases to simulate realistic inputs  
- Balanced class distribution for model training and testing  

---

## Feedforward Neural Network (FFNN)

A dense neural network was trained on the BERT embeddings. The FFNN used ReLU activations and softmax output to classify sentences. It achieved strong performance on clean data but showed reduced generalization on noisy variants.

---

## Logistic Regression and SVM

Two classic ML models were trained on BERT embeddings:

- **Logistic Regression**: Achieved 100% accuracy on the clean dataset but dropped to ~25% on noisy data.  
- **Support Vector Machine (SVM)**: Similar performance to logistic regression, showing that both models overfit to the clean structure and struggled with noise.

---

## Key Takeaways

- BERT embeddings significantly improved model understanding of slang context.
- Rule-based methods were effective but limited by vocabulary scope.
- All models performed poorly on noisy, real-world-like input, highlighting the need for domain adaptation.
- Linear models overfit to clean synthetic data and failed to generalize.

---

## Future Work

- Fine-tune BERT on a larger, real-world generational dataset  
- Scrape social platforms like Twitter, TikTok, and Reddit for training data  
- Add model explainability tools (e.g., SHAP, attention heatmaps)  
- Study slang evolution over time and its cross-generational shifts  

---

## How to Run

Launch the notebook:

```bash
jupyter notebook CS4395_NLP_Final_Project (1).ipynb
```

Or open in Google Colab.

---

## Requirements

```bash
pip install transformers pandas scikit-learn matplotlib
```

---

## Project Structure

| File                          | Description                                  |
|------------------------------|----------------------------------------------|
| `CS4395_NLP_Final_Project.ipynb` | Main notebook containing all code and outputs |
| `noisy_slang_dataset.csv`    | Synthetic dataset with slang and noise       |
| `README.md`                  | Project summary and methodology              |

