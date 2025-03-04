#!/usr/bin/env python3
"""
This script performs the following tasks:
1. Reads a CSV file ("results.csv") to generate a list of GitHub repository URLs.
2. Extracts Java methods from repositories using PyDriller and javalang.
3. Saves extracted methods to a CSV file.
4. Loads the CSV, performs several preprocessing steps (duplicate removal, filtering, tokenization).
5. Trains an n-gram language model, evaluates it (perplexity), and selects the best model.
6. Generates code completions based on the test set and saves the results to JSON files.
7. Prints a summary of the completions and model predictions.
    
Requirements:
    - pydriller
    - javalang
    - pygments
    - pandas
    - Other standard Python libraries
    
Note: Package installation and system update commands are commented out.
"""

# Uncomment these lines if running interactively and you need to install packages:
# import os
# os.system("pip install pydriller")
# os.system("pip install javalang")

# Uncomment these if you need to install git via apt (for Linux systems):
# os.system("add-apt-repository ppa:git-core/ppa")
# os.system("apt-get update")
# os.system("apt-get install git")

import os
import re
import csv
import json
import math
import random
import collections
import pandas as pd
from pydriller import Repository
import javalang
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from IPython.display import display  # You can remove this if not needed

# ---------------------- Step 1: Read Repository List ----------------------
df_res = pd.read_csv('results.csv')

repoList = []
for idx, row in df_res.iterrows():
    repoList.append("https://www.github.com/{}".format(row['name']))

print("Repository List (first 10):", repoList[0:10])

# ---------------------- Step 2: Java Methods Extraction Functions ----------------------
def extract_methods_from_java(code):
    """
    Extract methods from Java source code using javalang parser.
    
    Args:
        code (str): The Java source code.
    
    Returns:
        list: A list of tuples containing method names and their full source code.
    """
    methods = []
    try:
        tree = javalang.parse.parse(code)
        lines = code.splitlines()
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            method_name = node.name
            start_line = node.position.line - 1
            end_line = None
            if node.body:
                last_statement = node.body[-1]
                if hasattr(last_statement, 'position') and last_statement.position:
                    end_line = last_statement.position.line
            if end_line:
                method_code = "\n".join(lines[start_line:end_line+1])
            else:
                method_code = "\n".join(lines[start_line:])
            methods.append((method_name, method_code))
    except Exception as e:
        print(f"Error parsing Java code: {e}")
    return methods

def extract_methods_to_csv_from_master(repo_path, output_csv):
    """
    Extract methods from Java files in the master branch and append them to a CSV file.
    
    Args:
        repo_path (str): Path or URL to the Git repository.
        output_csv (str): Path to the output CSV file.
    """
    method_count = 0
    file_exists = os.path.exists(output_csv)
    with open(output_csv, mode='a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Commit Hash", "File Name", "Method Name", "Method Code", "Commit Link"])
        for commit in Repository(repo_path, only_in_branch="master").traverse_commits():
            print(f"Processing commit: {commit.hash}")
            for modified_file in commit.modified_files:
                if modified_file.filename.endswith(".java") and modified_file.source_code:
                    methods = extract_methods_from_java(modified_file.source_code)
                    for method_name, method_code in methods:
                        commit_link = f"{repo_path}/commit/{commit.hash}"
                        csv_writer.writerow([commit.hash, modified_file.filename, method_name, method_code, commit_link])
                        method_count += 1
                    print(f"Extracted {len(methods)} methods from {modified_file.filename} in commit {commit.hash}")
                    print(f"Total methods extracted so far: {method_count}")
    print(f"Finished extracting methods. Total methods extracted: {method_count}")

def extract_methods_to_csv(repo_path, output_csv):
    """
    Extract methods from Java files in a repository and append them to a CSV file.
    
    Args:
        repo_path (str): Path or URL to the Git repository.
        output_csv (str): Path to the output CSV file.
    """
    method_count = 0
    file_exists = os.path.exists(output_csv)
    with open(output_csv, mode='a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(["Branch Name", "Commit Hash", "File Name", "Method Name", "Method Code", "Commit Link"])
        branch_name = "master"
        for commit in Repository(repo_path, only_in_branch=branch_name).traverse_commits():
            print(f"Processing commit: {commit.hash}")
            for modified_file in commit.modified_files:
                if modified_file.filename.endswith(".java") and modified_file.source_code:
                    methods = extract_methods_from_java(modified_file.source_code)
                    for method_name, method_code in methods:
                        commit_link = f"{repo_path}/commit/{commit.hash}"
                        csv_writer.writerow([branch_name, commit.hash, modified_file.filename, method_name, method_code, commit_link])
                        method_count += 1
                    print(f"Extracted {len(methods)} methods from {modified_file.filename} in commit {commit.hash}")
                    print(f"Total methods extracted so far: {method_count}")
    print(f"Finished extracting methods. Total methods extracted: {method_count}")

# ---------------------- Example Usage of Method Extraction ----------------------
output_csv_file = "extracted_methods.csv"
for repo in repoList[0:20]:
    extract_methods_to_csv_from_master(repo, output_csv_file)
    print("Processed repo:", repo)

# ---------------------- Step 3: Preprocessing Functions ----------------------
def remove_duplicates(data):
    """Remove duplicate methods based on method content."""
    return data.drop_duplicates(subset="Method Code", keep="first")

def filter_ascii_methods(data):
    """Filter methods to include only those with ASCII characters."""
    return data[data["Method Code"].apply(lambda x: all(ord(char) < 128 for char in str(x)))]

def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    """Remove outliers based on method length."""
    method_lengths = data["Method Code"].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    return data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]

def remove_boilerplate_methods(data):
    boilerplate_patterns = [
        r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",
        r"\bget[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",
    ]
    boilerplate_regex = re.compile("|".join(boilerplate_patterns))
    return data[~data["Method Code"].apply(lambda x: bool(boilerplate_regex.search(str(x))))]

def remove_comments_from_dataframe(df, method_column, language):
    def remove_comments(code):
        lexer = get_lexer_by_name(language)
        tokens = lexer.get_tokens(code)
        return ''.join(token[1] for token in tokens if not (lambda t: t[0] in Token.Comment)(token))
    df["Method Code No Comments"] = df[method_column].apply(remove_comments)
    return df

def clean_and_tokenize(text):
    """Tokenize the text while removing tabs, newlines, and extra spaces."""
    text = text.strip()
    text = re.sub(r'[\t\n\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    lexer = get_lexer_by_name("java")
    tokens = [t[1].strip() for t in lexer.get_tokens(text) if t[1].strip() != '']
    return tokens

def tokenize_methods(df):
    df["Tokens"] = df["Method Code No Comments"].apply(clean_and_tokenize)
    df["Token Count"] = df["Tokens"].apply(len)
    return df

# ---------------------- Step 4: Load and Process Data ----------------------
data = pd.read_csv('extracted_methods.csv')
data = remove_duplicates(data)
data = filter_ascii_methods(data)
data = remove_outliers(data)
data = remove_boilerplate_methods(data)
data = remove_comments_from_dataframe(data, "Method Code", "java")
processed_data = tokenize_methods(data)

# If running as a script (not in Jupyter), you may want to use print() instead of display()
print(processed_data[['Method Code No Comments', 'Tokens', 'Token Count']].head())
print(processed_data["Token Count"].describe())

# ---------------------- Step 5: N-gram Model Functions ----------------------
def train_ngram_model(corpus, n):
    """
    Trains an n-gram model on the given corpus.
    
    Args:
        corpus (list of list of str): A list where each element is a list of tokens.
        n (int): The n value for the n-gram.
    
    Returns:
        ngram_counts, context_counts, vocabulary
    """
    ngram_counts = collections.Counter()
    context_counts = collections.Counter()
    vocabulary = set()
    for tokens in corpus:
        vocabulary.update(tokens)
        padded_tokens = ['<s>'] * (n - 1) + tokens + ['</s>']
        for i in range(len(padded_tokens) - n + 1):
            ngram = tuple(padded_tokens[i:i+n])
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
    log_prob_sum = 0.0
    M = 0
    for tokens in corpus:
        padded_tokens = ['<s>'] * (n - 1) + tokens + ['</s>']
        for i in range(len(padded_tokens) - n + 1):
            ngram = tuple(padded_tokens[i:i+n])
            prob = ngram_probability(ngram, ngram_counts, context_counts, vocab_size, smoothing)
            log_prob_sum += math.log(prob)
            M += 1
    return math.exp(-log_prob_sum / M) if M > 0 else float('inf')

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
        steps.append({
            "context": list(context),
            "predicted_token": next_token,
            "predicted_probability": candidates[next_token] / total_prob
        })
        if next_token == '</s>':
            break
        generated.append(next_token)
    return generated, steps

# ---------------------- Step 6: Prepare Corpus and Split Data ----------------------
all_methods = processed_data['Tokens'].tolist()
random.seed(42)
random.shuffle(all_methods)
n_total = len(all_methods)
n_train = int(0.8 * n_total)
n_eval = int(0.1 * n_total)
train_corpus = all_methods[:n_train]
eval_corpus = all_methods[n_train:n_train+n_eval]
test_corpus = all_methods[n_train+n_eval:]
print("Corpus sizes: Train = {}, Eval = {}, Test = {}".format(len(train_corpus), len(eval_corpus), len(test_corpus)))

# ---------------------- Step 7: Model Training, Evaluation, and Selection ----------------------
ngram_values = [3, 4, 5, 6, 7, 8, 9, 10]
results = {}
best_n = None
best_perplexity = float('inf')
best_model = None

for n in ngram_values:
    print(f"\nTraining {n}-gram model...")
    ngram_counts, context_counts, vocabulary = train_ngram_model(train_corpus, n)
    perplexity = compute_perplexity(eval_corpus, n, ngram_counts, context_counts, len(vocabulary))
    results[n] = perplexity
    print(f"n = {n} -> Evaluation Perplexity: {perplexity:.2f}")
    if perplexity < best_perplexity:
        best_perplexity = perplexity
        best_n = n
        best_model = (ngram_counts, context_counts, vocabulary)

print(f"\nBest model: {best_n}-gram with perplexity {best_perplexity:.2f}")

# ---------------------- Step 8: Code Completion on Test Set ----------------------
completions = []
detailed_steps = {}
ngram_counts, context_counts, vocabulary = best_model

for i, tokens in enumerate(test_corpus[:100]):
    initial_context = tokens[:best_n-1] if len(tokens) >= best_n - 1 else tokens
    ground_truth = tokens[len(initial_context):]
    generated, steps = generate_completion(initial_context, best_n, ngram_counts, context_counts, vocabulary)
    if len(generated) >= best_n:
        predicted_ngram = tuple(generated[-best_n:])
        pred_prob = ngram_probability(predicted_ngram, ngram_counts, context_counts, len(vocabulary))
    else:
        predicted_ngram = None
        pred_prob = None
    completions.append({
        "test_index": i,
        "initial_context": initial_context,
        "ground_truth_continuation": ground_truth,
        "predicted_completion": generated[len(initial_context):],
        "predicted_ngram": predicted_ngram,
        "predicted_probability": pred_prob
    })
    detailed_steps[f"test_index_{i}"] = steps

results_dict = {
    "model": {
        "n": best_n,
        "evaluation_perplexity": best_perplexity,
        "all_perplexities": results
    },
    "completions": completions
}

with open("results_student_model.json", "w", encoding="utf-8") as f:
    json.dump(results_dict, f, indent=4)

with open("detailed_steps.json", "w", encoding="utf-8") as f:
    json.dump(detailed_steps, f, indent=4)

print("\nGround Truth vs. Predicted Completions (First 5 Test Instances):")
for comp in completions[:5]:
    print(f"\nTest Index: {comp['test_index']}")
    print("Initial Context:")
    print(" ".join(comp["initial_context"]))
    print("Ground Truth Continuation:")
    print(" ".join(comp["ground_truth_continuation"]))
    print("Predicted Completion:")
    print(" ".join(comp["predicted_completion"]))

print("\nStep-by-step predictions for the first test instance:")
for step in detailed_steps.get("test_index_0", []):
    print(f"Context: {step['context']} -> Predicted: {step['predicted_token']} (Prob: {step['predicted_probability']:.4f})")

print("\nResults saved to 'results_student_model.json' and 'detailed_steps.json'")
    
if __name__ == "__main__":
    # The main execution block can be used to encapsulate functionality if needed.
    pass
