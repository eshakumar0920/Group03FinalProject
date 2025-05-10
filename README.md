# Generational Slang Classification using BERT

This project explores the classification of generational language — Boomer, Gen X, Millennial, and Gen Z — based on slang usage. It combines rule-based scoring with BERT embeddings and machine learning models to evaluate how well different approaches capture generational patterns in language.

## Project Contents

| File                           | Description                                      |
|--------------------------------|--------------------------------------------------|
| `CS4395_NLP_Final_Project.ipynb` | Main notebook containing all experiments         |
| `noisy_slang_dataset.csv`         | Synthetic dataset with typos and shared slang    |
| `README.md`                       | Project overview and methodology                 |

## Methodology

- **Dataset Generation**  
  Created a synthetic dataset of 20,000 sentences using slang from each generation. Added neutral sentences and typo/noise variants to simulate real-world ambiguity.

- **Preprocessing**  
  Used Hugging Face's BERT tokenizer and extracted `[CLS]` token embeddings for sentence representation.

- **Models Compared**
  - Rule-based scorer (dictionary matching)
  - Feedforward Neural Network (FFNN) using BERT embeddings
  - Logistic Regression
  - Support Vector Machine (SVM, Linear)

## Results Summary

### Clean Synthetic Dataset

| Model                | Accuracy |
|---------------------|----------|
| Rule-based Scorer   | ~88%     |
| FFNN + BERT         | ~83%     |
| Logistic Regression | 100%     |
| SVM (Linear)        | 100%     |

### Noisy/Realistic Dataset

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | ~25%     |
| SVM (Linear)        | ~25%     |

Models performed well on clean, separable data, but struggled with noisy, ambiguous input where slang overlaps across generations.

## Key Takeaways

- BERT embeddings were highly effective at distinguishing generational slang in clean data.
- On noisy input, performance dropped to near-random levels, revealing the difficulty of generalizing to real-world language variation.
- The rule-based approach is fast and interpretable but limited in coverage.

## Future Work

- Fine-tune BERT directly on generational classification
- Use real-world social media data (e.g., Reddit, Twitter, TikTok)
- Analyze slang evolution over time
- Explore model interpretability (e.g., attention visualizations)

## Requirements

```bash
pip install transformers pandas scikit-learn matplotlib
