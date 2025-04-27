# Install libraries (only needed once)
#!pip install transformers nltk scikit-learn torch

# Import libraries
import nltk
from nltk.corpus import wordnet
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK data (WordNet only)
nltk.download('wordnet')

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Helper Functions
def get_contextual_embedding(word, sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state.squeeze(0)
    token_ids = inputs["input_ids"][0]

    decoded_tokens = [tokenizer.decode([id]) for id in token_ids]

    for i, tok in enumerate(decoded_tokens):
        if word.lower() in tok.lower():
            return token_embeddings[i].numpy()
    return None

# Detect if slang is used based on similarity
def is_used_slangy(word, sentence, threshold=0.5):
    context_embed = get_contextual_embedding(word, sentence)
    if context_embed is None:
        return False

    standard_sentence = f"The {word} is very common."
    standard_embed = get_contextual_embedding(word, standard_sentence)
    if standard_embed is None:
        return False

    similarity = cosine_similarity([context_embed], [standard_embed])[0][0]
    return similarity < threshold

# Main code
slang_dict = {
    'fr': 'for real',
    'fit': 'outfit',
    'fire': 'cool or amazing'
}

# Example sentence
sentence = "That fit is fire fr"

# Tokenize sentence
inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.tokenize(sentence)

# Build standard words set
standard_words = set(lemma.name().lower() for synset in wordnet.all_synsets() for lemma in synset.lemmas())

# Add basic English words manually to cover common gaps
basic_english_words = set([
    "that", "this", "is", "are", "was", "were", "am", "be", "being", "been", "have", "has", "had",
    "do", "does", "did", "the", "a", "an", "in", "on", "at", "by", "for", "with", "about", "against", "between"
])

# Classify each token
results = []

for word in tokens:
    lw = word.lower().strip()  # Lowercase and clean the word
    in_standard = lw in standard_words or lw in basic_english_words
    in_slang = lw in slang_dict

    if in_standard and in_slang:
        if is_used_slangy(word, sentence):
            box = "Slang Use Detected via BERT"
        else:
            box = "Standard Use Detected via BERT"
    elif in_standard:
        if is_used_slangy(word, sentence):
            box = "Slang Use Detected via BERT"
        else:
            box = "Standard Word"
    elif in_slang:
        box = "Slang Only"
    else:
        box = "Unknown/Not a Word"

    results.append((word, box))

# Output results
for word, category in results:
    print(f"{word}: {category}")