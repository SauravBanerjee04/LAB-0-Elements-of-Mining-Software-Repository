#!/usr/bin/env python3
"""
Script: train_model.py

This script reads a CSV file of Java methods (with a "Method Code" column), preprocesses the data 
(by removing duplicates, filtering non-ASCII characters, outlier and boilerplate removal, and tokenizing),
splits the corpus into training, evaluation, and test sets, then trains several N-gram models (with different N values),
selects the best model based on evaluation perplexity, and generates predictions on 100 test examples.
The predictions are saved in a JSON file with the structure:
  {
    "0": [ ["pred_token1", "0.3123"], ["pred_token2", "0.9845"], ... ],
    "1": [ ... ],
    ...
    "99": [ ... ]
  }

Additionally, the script saves the best-performing model as a pickle file ("trained_model.pkl").

Usage:
For a student corpus (e.g., the CSV produced by extract_methods.py):
  python train_model.py --input extracted_methods_student.csv --mode student

For a teacher corpus:
  python train_model.py --input extracted_methods_teacher.csv --mode teacher
"""

import os, re, csv, json, math, random, collections, argparse, pickle
import pandas as pd
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

# ---------------------- Preprocessing Functions ----------------------
def remove_duplicates(data):
    return data.drop_duplicates(subset="Method Code", keep="first")

def filter_ascii_methods(data):
    return data[data["Method Code"].apply(lambda x: all(ord(char) < 128 for char in str(x)))]

def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    lengths = data["Method Code"].apply(len)
    lower = lengths.quantile(lower_percentile / 100)
    upper = lengths.quantile(upper_percentile / 100)
    return data[(lengths >= lower) & (lengths <= upper)]

def remove_boilerplate_methods(data):
    patterns = [r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{", r"\bget[A-Z][a-zA-Z0-9_]*\(.*\)\s*{"]
    regex = re.compile("|".join(patterns))
    return data[~data["Method Code"].apply(lambda x: bool(regex.search(str(x))))]

def clean_and_tokenize(text):
    text = text.strip()
    text = re.sub(r'[\t\n\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    lexer = get_lexer_by_name("java")
    tokens = [t[1].strip() for t in lexer.get_tokens(text) if t[1].strip() != '']
    return tokens

def tokenize_methods(df, column="Method Code"):
    df["Tokens"] = df[column].apply(clean_and_tokenize)
    df["Token Count"] = df["Tokens"].apply(len)
    return df

# ---------------------- N-gram Model Functions ----------------------
def train_ngram_model(corpus, n):
    ngram_counts = collections.Counter()
    context_counts = collections.Counter()
    vocabulary = set()
    for tokens in corpus:
        vocabulary.update(tokens)
        padded = ['<s>'] * (n - 1) + tokens + ['</s>']
        for i in range(len(padded) - n + 1):
            ngram = tuple(padded[i:i+n])
            context = ngram[:-1]
            ngram_counts[ngram] += 1
            context_counts[context] += 1
    return ngram_counts, context_counts, vocabulary

def ngram_probability(ngram, ngram_counts, context_counts, vocab_size, smoothing=1):
    context = ngram[:-1]
    count_ngram = ngram_counts.get(ngram, 0)
    count_context = context_counts.get(context, 0)
    return (count_ngram + smoothing) / (count_context + smoothing * vocab_size)

def compute_perplexity(corpus, n, ngram_counts, context_counts, vocab_size, smoothing=1):
    log_prob = 0.0
    total = 0
    for tokens in corpus:
        padded = ['<s>'] * (n - 1) + tokens + ['</s>']
        for i in range(len(padded) - n + 1):
            ngram = tuple(padded[i:i+n])
            prob = ngram_probability(ngram, ngram_counts, context_counts, vocab_size, smoothing)
            log_prob += math.log(prob)
            total += 1
    return math.exp(-log_prob / total) if total > 0 else float('inf')

def generate_completion(initial_tokens, n, ngram_counts, context_counts, vocabulary, max_tokens=20, smoothing=1):
    generated = initial_tokens.copy()
    steps = []
    for _ in range(max_tokens):
        context = tuple((['<s>'] * max(0, n - 1 - len(generated)) + generated)[- (n - 1):])
        candidates = {}
        for token in vocabulary.union({'</s>'}):
            ngram = context + (token,)
            prob = ngram_probability(ngram, ngram_counts, context_counts, len(vocabulary), smoothing)
            if prob > 0:
                candidates[token] = prob
        if not candidates:
            break
        total_prob = sum(candidates.values())
        tokens, probs = zip(*[(t, p / total_prob) for t, p in candidates.items()])
        next_token = random.choices(tokens, weights=probs)[0]
        steps.append((next_token, f"{candidates[next_token] / total_prob:.4f}"))
        if next_token == '</s>':
            break
        generated.append(next_token)
    return generated, steps

def run_pipeline(corpus, output_filename):
    # Shuffle and split corpus
    random.seed(42)
    random.shuffle(corpus)
    total = len(corpus)
    train_corpus = corpus[:int(0.8 * total)]
    eval_corpus = corpus[int(0.8 * total):int(0.9 * total)]
    test_corpus = corpus[int(0.9 * total):]
    print(f"Corpus sizes: Train={len(train_corpus)}, Eval={len(eval_corpus)}, Test={len(test_corpus)}")
    
    # Train several N-gram models
    ngram_values = [3, 4, 5, 6, 7, 8, 9, 10]
    best_n = None
    best_perplexity = float('inf')
    best_model = None
    results = {}
    for n in ngram_values:
        print(f"\nTraining {n}-gram model...")
        ngram_counts, context_counts, vocabulary = train_ngram_model(train_corpus, n)
        perp = compute_perplexity(eval_corpus, n, ngram_counts, context_counts, len(vocabulary))
        results[n] = perp
        print(f"n = {n}: Evaluation Perplexity = {perp:.2f}")
        if perp < best_perplexity:
            best_perplexity = perp
            best_n = n
            best_model = (ngram_counts, context_counts, vocabulary)
    print(f"\nBest model: {best_n}-gram with perplexity {best_perplexity:.2f}")
    
    # Save the best model as a pickle file
    model_data = {
        "n": best_n,
        "ngram_counts": best_model[0],
        "context_counts": best_model[1],
        "vocabulary": best_model[2],
        "evaluation_perplexity": best_perplexity,
        "all_perplexities": results
    }
    with open("trained_model.pkl", "wb") as pkl_file:
        pickle.dump(model_data, pkl_file)
    print("Best model saved to 'trained_model.pkl'")
    
    # Generate predictions for 100 test instances
    predictions = {}
    ngram_counts, context_counts, vocabulary = best_model
    for i, tokens in enumerate(test_corpus[:100]):
        initial_context = tokens[:best_n-1] if len(tokens) >= best_n-1 else tokens
        _, steps = generate_completion(initial_context, best_n, ngram_counts, context_counts, vocabulary)
        predictions[str(i)] = steps
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)
    print(f"\nPredictions saved to '{output_filename}'")

def main():
    parser = argparse.ArgumentParser(description="Train an N-gram model on a corpus of Java methods and output predictions.")
    parser.add_argument("--input", required=True, help="Input CSV file with a 'Method Code' column")
    parser.add_argument("--mode", choices=["student", "teacher"], default="student",
                        help="Corpus mode: 'student' or 'teacher'")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    # Preprocess the corpus
    df = remove_duplicates(df)
    df = filter_ascii_methods(df)
    df = remove_outliers(df)
    df = remove_boilerplate_methods(df)
    df = tokenize_methods(df, column="Method Code")
    corpus = df["Tokens"].tolist()
    
    output_filename = "results_student_model.json" if args.mode == "student" else "results_teacher_model.json"
    run_pipeline(corpus, output_filename)

if __name__ == "__main__":
    main()
