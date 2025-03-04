#!/usr/bin/env python3
"""
Script: evaluate_model.py

This script reads a CSV file of Java methods (with a "Method Code" column), tokenizes the methods,
trains N-gram models for various N values on the entire corpus, and prints out the perplexity values.
This can help you report intrinsic evaluation results.
"""

import os, math, random, collections, argparse, pandas as pd, re
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

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

def main():
    parser = argparse.ArgumentParser(description="Evaluate N-gram model perplexity on a corpus of Java methods")
    parser.add_argument("--input", required=True, help="Input CSV file with a 'Method Code' column")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    df = tokenize_methods(df, column="Method Code")
    corpus = df["Tokens"].tolist()
    
    ngram_values = [3, 4, 5, 6, 7, 8, 9, 10]
    best_n = None
    best_perplexity = float('inf')
    for n in ngram_values:
        ngram_counts, context_counts, vocabulary = train_ngram_model(corpus, n)
        perp = compute_perplexity(corpus, n, ngram_counts, context_counts, len(vocabulary))
        print(f"n = {n}: Perplexity = {perp:.2f}")
        if perp < best_perplexity:
            best_perplexity = perp
            best_n = n
    print(f"\nBest model is {best_n}-gram with perplexity {best_perplexity:.2f}")

if __name__ == "__main__":
    main()
