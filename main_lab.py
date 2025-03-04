#!/usr/bin/env python3
"""
Main Lab Script for CSCI 420/520: GenAI for SD

This script uses helper functions from extract_methods.py and train_model.py to:
  1. Load a corpus of Java methods from either a CSV file (student mode) or a plain text file (teacher mode).
  2. Preprocess and tokenize the methods.
  3. Split the data into training, evaluation, and test sets.
  4. Train multiple N-gram models, select the best model (based on perplexity over the eval set),
     and generate code completion predictions for 100 test examples.
  5. Save the predictions in a JSON file and the best model as a pickle file.
  
Usage:
  python main_lab.py --input extracted_methods_student.csv --mode student
  python main_lab.py --input training.txt --mode teacher
"""

import argparse
import pandas as pd

# Import helper functions from your existing modules.
# Adjust the names if necessary.
from train_model import (
    remove_duplicates,
    filter_ascii_methods,
    remove_outliers,
    remove_boilerplate_methods,
    tokenize_methods,
    run_pipeline
)

def load_corpus(input_file, mode):
    """
    Loads the corpus based on the mode.
      - student mode: expects a CSV file with a "Method Code" column.
      - teacher mode: expects a plain text file where each line is a pre-tokenized method.
    Returns:
      - For student: a pandas DataFrame.
      - For teacher: a list of strings (each string is a tokenized method line).
    """
    if mode == "student":
        df = pd.read_csv(input_file)
        return df
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines

def prepare_corpus(df_or_lines, mode):
    """
    Processes the input data and returns a corpus.
      - In student mode, the DataFrame is preprocessed (duplicate removal, ASCII filtering,
        outlier & boilerplate removal, and tokenization) and returns a list of token lists.
      - In teacher mode, each line (assumed already tokenized) is split into a list of tokens.
    """
    if mode == "student":
        # Preprocessing for student data using helper functions from train_model.py.
        df = remove_duplicates(df_or_lines)
        df = filter_ascii_methods(df)
        df = remove_outliers(df)
        df = remove_boilerplate_methods(df)
        df = tokenize_methods(df, column="Method Code")
        corpus = df["Tokens"].tolist()
    else:
        # For teacher mode, each line is assumed to be pre-tokenized (tokens separated by whitespace)
        corpus = [line.split() for line in df_or_lines]
    return corpus

def main():
    parser = argparse.ArgumentParser(description="Main script for training and evaluating an N-gram model for code completion.")
    parser.add_argument("--input", required=True,
                        help="Input corpus file. For student mode: CSV with 'Method Code' column; for teacher mode: plain text file with each line a tokenized method.")
    parser.add_argument("--mode", choices=["student", "teacher"], required=True,
                        help="Corpus mode: 'student' or 'teacher'")
    args = parser.parse_args()
    
    # Load raw data.
    data = load_corpus(args.input, args.mode)
    # Prepare the corpus (list of token lists).
    corpus = prepare_corpus(data, args.mode)
    
    # Determine output JSON filename.
    output_filename = "results_student_model.json" if args.mode == "student" else "results_teacher_model.json"
    
    # Run the training/evaluation pipeline using the helper function from train_model.py.
    run_pipeline(corpus, output_filename)

if __name__ == "__main__":
    main()
