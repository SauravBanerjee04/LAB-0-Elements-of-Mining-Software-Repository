#!/usr/bin/env python3
"""
Script: extract_methods.py

For student mode:
  Reads "results.csv" (which contains repository names), iterates over the repositories,
  extracts Java methods using PyDriller and javalang, and writes the results to an output CSV 
  (e.g., "extracted_methods_student.csv").

For teacher mode:
  Reads a file (e.g., "training.txt") where each line is a pre‑tokenized Java method and writes 
  each method as a row to an output CSV (e.g., "extracted_methods_teacher.csv").
"""

import sys
import os
import csv
import argparse
import pandas as pd
from pydriller import Repository
import javalang

def extract_methods_from_java(code, max_depth=3000):
    """
    Extract methods from Java source code using javalang parser.
    Temporarily increases the recursion limit to 'max_depth' to handle deep ASTs.
    If a RecursionError occurs, we skip that file and move on.
    """
    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max_depth)  # Increase recursion limit
    
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
    except RecursionError:
        # If we hit the recursion limit, skip parsing this file
        print("Skipping a file due to exceeding recursion depth.")
    except Exception as e:
        # For other exceptions, just print and skip
        print(f"Error parsing Java code: {e}")
    finally:
        # Restore the original recursion limit
        sys.setrecursionlimit(original_limit)
    
    return methods

def process_student_csv(input_csv, output_csv):
    """Process a student results CSV by extracting methods from each repository."""
    df = pd.read_csv(input_csv)
    repoList = []
    for idx, row in df.iterrows():
        repoList.append("https://www.github.com/{}".format(row['name']))
    # Print only first 20
    print("Processing repositories (first 20):", repoList[:20])
    
    method_count = 0
    with open(output_csv, mode='w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Commit Hash", "File Name", "Method Name", "Method Code", "Commit Link"])
        # Process only first 20 repositories
        for repo in repoList[:20]:
            print(f"Processing repo: {repo}")
            for commit in Repository(repo, only_in_branch="master").traverse_commits():
                for modified_file in commit.modified_files:
                    if modified_file.filename.endswith(".java") and modified_file.source_code:
                        methods = extract_methods_from_java(modified_file.source_code)
                        for mname, mcode in methods:
                            commit_link = f"{repo}/commit/{commit.hash}"
                            writer.writerow([commit.hash, modified_file.filename, mname, mcode, commit_link])
                            method_count += 1
    print(f"Finished extracting methods. Total methods extracted: {method_count}")

def process_teacher_txt(input_txt, output_csv):
    """Process a teacher text file where each line is a pre-tokenized Java method."""
    with open(input_txt, "r", encoding="utf-8") as fin, open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["Method Code"])
        for line in fin:
            line = line.strip()
            if line:
                writer.writerow([line])
    print(f"Processed teacher file; output written to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Extract Java methods from a source file.")
    parser.add_argument("--input", required=True, help="Input filename (results.csv for student or training.txt for teacher)")
    parser.add_argument("--output", required=True, help="Output CSV filename")
    parser.add_argument("--mode", choices=["student", "teacher"], required=True, help="Mode: 'student' or 'teacher'")
    args = parser.parse_args()
    
    if args.mode == "student":
        process_student_csv(args.input, args.output)
    else:
        process_teacher_txt(args.input, args.output)

if __name__ == "__main__":
    main()
