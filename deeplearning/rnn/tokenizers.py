import re

# def my_tokenizer_word(text):
#     text = re.sub(r'<[^>]*>', '', text)
#     emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
#     text = re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
#     tokenized = text.split()
#     return tokenized

# def my_tokenizer_number(text, vocab, max_len=200):
#     tokens = my_tokenizer_word(text)
#     indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
#     # Truncate if too long
#     if len(indices) > max_len:
#         indices = indices[:max_len]
#     return indices

def my_tokenizer_word(text):
    if isinstance(text, list):
        tokenized = []
        for t in text:
            t = re.sub(r'<[^>]*>', '', t)
            emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', t.lower())
            t = re.sub(r'[\W]+', ' ', t.lower()) + ' '.join(emoticons).replace('-', '')
            # tokenized.extend(t.split())
            tokenized.append(t.split())
        return tokenized
        
    else:
        text = re.sub(r'<[^>]*>', '', text)
        emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
        text = re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
        tokenized = text.split()
        return tokenized



def my_tokenizer_number(text, vocab, max_len=200):
    if isinstance(text, list):
        all_indices = []
        for t in text:
            tokens = my_tokenizer_word(t)
            indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
            # Truncate if too long
            if len(indices) > max_len:
                indices = indices[:max_len]
            all_indices.append(indices)
        return all_indices
    else:
        tokens = my_tokenizer_word(text)
        indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
        # Truncate if too long
        if len(indices) > max_len:
            indices = indices[:max_len]
        return indices

if __name__ == "__main__":
    sample_text = "I am in tokinizer"
    print(my_tokenizer_word(sample_text))
