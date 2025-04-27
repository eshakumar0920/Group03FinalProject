# Example slang dictionary
slang_dict = {
    'fr': 'for real',
    'fit': 'outfit',
    'fire': 'cool or amazing'
}

# Example sentence
sentence = "That fit is fire fr"

# Tokenize
inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.tokenize(sentence)


# Load standard dictionary
standard_words = set(lemma.name().lower() for synset in wordnet.all_synsets() for lemma in synset.lemmas())


# Classify tokens
results = []

for word in tokens:
    lw = word.lower()
    in_standard = lw in standard_words
    in_slang = lw in slang_dict

    if in_standard and in_slang:
        box = "Ambiguous (Standard & Slang)"
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

# Show results
for word, category in results:
    print(f"{word}: {category}")
