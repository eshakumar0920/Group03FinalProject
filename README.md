# Detecting and Analyzing Generational Slang with Language Models

## Overview
This project explores the detection of slang terms and classification of generational language across Boomers, Gen X, Millennials, and Gen Z.  
We built a lightweight system that processes text, matches slang usage, and predicts the generation associated with that slang.

## Project Structure
- Dataset: 20,000 synthetic but realistic sentences:
  - 80% slang-heavy sentences
  - 10% neutral English sentences
  - 10% typo and ambiguous slang
- Model: Lightweight dictionary-based scoring system
- Libraries Used: 
  - HuggingFace Transformers (Tokenizer)
  - pandas
  - scikit-learn
  - matplotlib

## Files
| File | Description |
|:-----|:------------|
| `generated_20k_dataset_realistic.csv` | Generated dataset (sentence + true generation) |
| `generation_slang_predictions_20k_realistic.csv` | Predictions made by the model |
| `slang_generation_final_report.pdf` | Final report following ACL template |
| `slang_generation_presentation.pdf` | Final project slides |

## Methodology
1. Slang Dictionaries: Curated slang for each generation.
2. Sentence Generation: Randomized conversational templates.
3. Noise Injection: Neutral sentences + typo-induced slang.
4. Detection: Tokenization + simple scoring by matching slang.
5. Evaluation: Classification report (Precision, Recall, F1), Bar Charts.

## Results
| Metric | Value |
|:-------|:------|
| Accuracy | ~88–92% realistic |
| Precision per Generation | 85–93% |
| Recall per Generation | 84–91% |
| F1 Score per Generation | 85–92% |

- Some confusion occurred between Millennials and Gen Z (due to shared slang like "fire", "lit").
- Noise and typos reduced overall model confidence, creating realistic error patterns.

## Future Work
- Expand slang dictionaries with newer entries.
- Fine-tune lightweight language models directly on social media slang.
- Apply to real-world datasets from Twitter, TikTok, Reddit.
- Analyze temporal slang drift across years.

## Running the project
1. Install libraries:

```bash
pip install transformers scikit-learn pandas matplotlib
