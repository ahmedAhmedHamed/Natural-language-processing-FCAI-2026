from preprocessing import run_preprocessing

if __name__ == '__main__':
    sentences = "The quick brown fox jumps over the lazy dog, and the dog barked loudly!"
    tokens = run_preprocessing(sentences)
    print(tokens)
    # model = build n gram model (tokens)
    # generated_sentences = generate sentences (model)
