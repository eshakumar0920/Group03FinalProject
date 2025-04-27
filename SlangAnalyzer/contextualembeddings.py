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
