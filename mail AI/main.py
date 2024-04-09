import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import time
import tkinter as tk
from tkinter import filedialog

# Load the dataset
def load_dataset():
    try:
        df = pd.read_csv('spam.csv', header=None, names=['label', 'message'])
    except FileNotFoundError:
        print("Error: File 'spam.csv' not found.")
        exit(1)
    return df

# Preprocess the data
def preprocess_data(df):
    df['message'] = df['message'].str.lower()
    df['message'] = df['message'].str.replace(r'[^\w\s]', '')
    return df

# Train the classifier
def train_classifier(X_train, y_train):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf

# debug
def print_evaluation_metrics(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = (conf_matrix[0,0] + conf_matrix[1,1]) / (conf_matrix.sum())
    precision = conf_matrix[1,1] / (conf_matrix[1,0] + conf_matrix[1,1])
    recall = conf_matrix[1,1] / (conf_matrix[0,1] + conf_matrix[1,1])
    f1 = 2 * precision * recall / (precision + recall)

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")

# Browse file and update input textbox
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_contents = file.read()
        input_textbox.delete("1.0", tk.END)
        input_textbox.insert(tk.END, file_contents)


# Process input text
def process_input():
    email = input_textbox.get("1.0", tk.END)
    email = email.lower().replace(r'[^\w\s]', '')
    
    if not email.strip():
        console_text.delete("1.0", tk.END)
        console_text.insert(tk.END, "Error: Empty email. Please enter a valid email.", "red")
        return
    #start time
    start_time = time.time()
    email_vector = vectorizer.transform([email])
    try:
        email_prediction, email_prob = clf.predict(email_vector), clf.predict_proba(email_vector)
    except ValueError:
        console_text.delete("1.0", tk.END)
        console_text.insert(tk.END, "Error: Unable to make a prediction.", "red")
        return

    elapsed_time = (time.time() - start_time) * 1000
    if len(email_prob) == 0:
        console_text.delete("1.0", tk.END)
        console_text.insert(tk.END, "Error: Probability calculation failed.", "red")
        return

    if email_prediction == 'ham':
        console_text.delete("1.0", tk.END)
        console_text.insert(tk.END, f"Your email is NOT SPAM ({email_prob[0][1] * 100:.6f}%)\nTime ran: {elapsed_time:.6f} ms")
    else:
        console_text.delete("1.0", tk.END)
        console_text.insert(tk.END, f"Your email is SPAM ({email_prob[0][1] * 100:.6f}%)\nTime ran: {elapsed_time:.6f} ms")

# Main function
def main():
    df = load_dataset()
    df = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    global vectorizer
    vectorizer = CountVectorizer()
    global clf
    clf = train_classifier(vectorizer.fit_transform(X_train), y_train)

    y_pred = clf.predict(vectorizer.transform(X_test))
    print_evaluation_metrics(y_test, y_pred)

    root = tk.Tk()
    root.title("Naive Bayes Project")

    frame = tk.Frame(root, bg='#f0f0f0')
    frame.pack(padx=10, pady=10)

    global input_textbox
    input_textbox = tk.Text(frame, height=10, width=50, bg='#ffffff', bd=1, font=('Arial', 12))
    input_textbox.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

    global console_text
    console_text = tk.Text(frame, height=4, width=50, bg='#ffffff', bd=1, font=('Arial', 12))
    console_text.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
    console_text.tag_configure("red", foreground="red")
    browse_button = tk.Button(frame, text="Browse File", width=20, bg='#ffffff', fg='black', font=('Arial', 12), command=browse_file)
    browse_button.grid(row=2, column=1, padx=2, pady=5)

    process_button = tk.Button(frame, text="Enter", width=20, bg='#ffffff', fg='black', font=('Arial', 12), command=process_input)
    process_button.grid(row=2, column=0, padx=2, pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()


