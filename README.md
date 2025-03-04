# **GenAI for SD – N-gram Code Completion Lab**

1. **Corpus Construction & Preprocessing:**
    - Extract Java methods from GitHub repositories (student data) or use an instructor-provided corpus (teacher data).
    - Preprocess the corpus by removing duplicates, filtering non-ASCII characters, eliminating outlier or boilerplate methods, and tokenizing the Java code.
2. **Model Training & Evaluation:**
    - Train several N‑gram models (varying the context/window size).
    - Select the best model based on evaluation set perplexity.
    - Generate code completion predictions for 100 test methods.
    - Save the best model as a pickle file.

## **Repository Files**

- **extract_methods.py  
    **Extracts Java methods from a CSV file (for student data) or from a text file (for teacher data) and saves them to a CSV file.
- **train_model.py  
    **Preprocesses the input corpus, splits it into train/eval/test sets, trains multiple N‑gram models, selects the best one, generates predictions for 100 test examples, and saves the model as a pickle file.
- **evaluate_model.py  
    **(Optional) Evaluates different N‑gram configurations on the corpus and prints out perplexity values.
- **main_lab.py (or Lab0.py)  
    **A “clean” main script that reuses helper functions from the other files to run the complete lab pipeline.
- **training.txt  
    **A sample teacher corpus (each line is a pre‑tokenized Java method).
- **results.csv  
    **(Not included in the repository) – A CSV file containing repository names for student data.

## **Environment Setup**

### **Prerequisites**

- **Python 3.8+**
- **Git**

### **Setting Up on macOS or Windows**

**Clone the Repository:  
**bash  
CopyEdit  
git clone &lt;repository_url&gt;

cd &lt;repository_folder&gt;

**Create and Activate a Virtual Environment:  
**On **macOS/Linux**:  
bash  
CopyEdit  
python3 -m venv venv

source venv/bin/activate

On **Windows**:  
bash  
CopyEdit  
python -m venv venv

venv\\Scripts\\activate

**Install Dependencies:  
**Install the required packages using pip:  
bash  
CopyEdit  
pip install pandas pygments javalang pydriller
**Alternatively, use the included requirements list.
pip install -r requirements.txt

## **Generating the Extracted Methods CSV**

### **For Student Data**

1. Place your results.csv (a CSV containing repository names with a column named **name**) in the repository folder.

Run the following command to generate extracted_methods_student.csv:  
bash  
CopyEdit  
python extract_methods.py --input results.csv --output extracted_methods_student.csv --mode student

1. This script will iterate through the repository names listed in results.csv, extract Java methods using PyDriller and javalang, and output a CSV with columns: _Commit Hash, File Name, Method Name, Method Code, Commit Link_.

### **For Teacher Data**

1. Ensure you have your training.txt file (each line is a pre-tokenized Java method).

Run the following command to generate extracted_methods_teacher.csv:  
bash  
CopyEdit  
python extract_methods.py --input training.txt --output extracted_methods_teacher.csv --mode teacher

1. This will read each line from training.txt and write it as a row in the CSV (with a single column **Method Code**).

## **Running the Lab**

There are two main modes: **student** and **teacher**. Both modes run the same pipeline – the only difference is the input corpus.

### **Running the Student Version**

Use the CSV file generated from your student extraction (e.g., extracted_methods_student.csv).

bash

CopyEdit

python main_lab.py --input extracted_methods_student.csv --mode student

This command will:

- Load and preprocess the CSV (removing duplicates, filtering, tokenizing, etc.).
- Split the data into training (80%), evaluation (10%), and test (10%) sets.
- Train several N‑gram models (with different context sizes).
- Select the best-performing model based on evaluation perplexity.
- Generate predictions for 100 test Java methods and save them to results_student_model.json.
- Save the best model as trained_model.pkl.

### **Running the Teacher Version**

Use your instructor-provided text file (e.g., training.txt or the CSV extracted_methods_teacher.csv if already converted).

bash

CopyEdit

python main_lab.py --input training.txt --mode teacher

This command will:

- Load the plain text file and split each line into tokens.
- Follow the same pipeline as above.
- Save the predictions to results_teacher_model.json and the model as trained_model.pkl.

_Note: If you converted the teacher file to CSV using extract_methods.py, then use that CSV file with mode teacher._

## **Additional Evaluation**

If you want to inspect the perplexity of various N‑gram configurations on your corpus, run:

bash

CopyEdit

python evaluate_model.py --input extracted_methods_student.csv

Replace extracted_methods_student.csv with your teacher CSV if needed.

## **Summary**

1. **Extract Methods:  
    **Generate your CSV files using extract_methods.py (for student and/or teacher).(Either python extract_methods.py --input results.csv --output extracted_methods_student.csv --mode student or python extract_methods.py --input training.txt --output extracted_methods_teacher.csv --mode teacher)
2. **Train & Evaluate:  
    **Run main_lab.py (or Lab0.py) with the appropriate input file and mode to train the N‑gram model, generate predictions, and save the model.  (Either python main_lab.py --input extracted_methods_student.csv --mode student or python main_lab.py --input training.txt --mode teacher)
