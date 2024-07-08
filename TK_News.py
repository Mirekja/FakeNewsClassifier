# Projekt: Klassifikation von Satire/ Fake News
# im Modul: Machine Learning bei Andreas Heß
# Author des Codes: Mirco Jablonski
# Quellen: https://www.datacamp.com/cheat-sheet/scikit-learn-cheat-sheet-python-machine-learning, https://www.datacamp.com/cheat-sheet/matplotlib-cheat-sheet-plotting-in-python

import pandas as pd
import os
import csv
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import re
import arff
import numpy as np
import matplotlib.pyplot as plt
import nltk

# NLTK-Daten herunterladen
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Zusätzliche Stop-Wörter
custom_stop_words = ['postellion', 'anzeige', 'anzeig', 'titletext']

class NewsClassifier:
    def __init__(self):
        # Kombinierte Liste von Stop-Wörtern
        preprocessed_stop_words = [self._preprocess_word(word) for word in custom_stop_words]
        stop_words = stopwords.words('german') + preprocessed_stop_words
        self.vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 3))
        """
        Bitte die gewünschten Algorithmus einkommentieren und die unerwünschten auskommentieren
        """
        #self.clf = MultinomialNB()  # Naive Bayes
        self.clf = svm.SVC(kernel='linear')  # SVM
        #self.clf = LinearSVC()  # SVC

    def _preprocess_word(self, word):
        # Tokenisierung/Normalisierung von Stop-Wörtern
        return re.sub(r'\W+', '', word.lower())

    def _preprocess_text(self, text):
        # Satzzeichen entfernen und Groß- und Kleinschreibung normalisieren
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()

        # Tokenisierung und Lemmatisierung
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Entferne spezifische Wörter
        tokens = [word for word in tokens if word not in custom_stop_words]

        # Wortstammkürzung
        stemmer = SnowballStemmer('german')
        tokens = [stemmer.stem(word) for word in tokens]

        # Verbinde die tokens wieder zu einem Text
        processed_text = ' '.join(tokens)
        return processed_text

    def train(self, X_train, y_train):
        X_train_counts = self.vectorizer.fit_transform(X_train)
        self.clf.fit(X_train_counts, y_train)

    def evaluate(self, X_test, y_test):
        X_test_counts = self.vectorizer.transform(X_test)
        y_pred = self.clf.predict(X_test_counts)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return accuracy, precision, recall, f1, cm

    def classify(self, text, threshold=0):
        text = self._preprocess_text(text)
        text_counts = self.vectorizer.transform([text])
        probs = self.clf.decision_function(text_counts)
        prediction = 1 if probs[1] >= threshold else 0
        return prediction

def train_in_steps(classifier, steps, X_test, y_test):
    for step in steps:
        fake_news_path = f'Trainingsdaten/{step}%/FNT_{step}'
        true_news_path = f'Trainingsdaten{step}%/TNT_{step}'
        X_train_new, _, y_train_new, _ = load_data(fake_news_path, true_news_path)

        if not hasattr(classifier, 'X_train') or not hasattr(classifier, 'y_train'):
            classifier.X_train, classifier.y_train = X_train_new, y_train_new
        else:
            classifier.X_train = pd.concat([classifier.X_train, X_train_new])
            classifier.y_train = pd.concat([classifier.y_train, y_train_new])

        classifier.train(classifier.X_train, classifier.y_train)

        # Speichern des Klassifikators nach jedem Schritt
        step_save_path = f'{step}%'
        if not os.path.exists(step_save_path):
            os.makedirs(step_save_path)

        # Auswertung des Klassifikators nach jedem Schritt
        accuracy, precision, recall, f1, cm = classifier.evaluate(X_test, y_test)
        print(f'Accuracy after {step}% training: {accuracy:.4f}')
        print(f'Precision after {step}% training: {precision:.4f}')
        print(f'Recall after {step}% training: {recall:.4f}')
        print(f'F1 Score after {step}% training: {f1:.4f}')
        print('Confusion Matrix after update:')
        print(cm)

def load_data(fake_news_path, true_news_path):
    fake_news = load_csv(fake_news_path, 0)
    true_news = load_csv(true_news_path, 1)
    data = pd.concat([fake_news, true_news], ignore_index=True)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_csv(folder_path, label):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = []
    classifier = NewsClassifier()  # Instanz der NewsClassifier-Klasse erstellen
    for file in files:
        with open(file, 'r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if len(row) > 0:  # Überprüfe, ob die Zeile nicht leer ist
                    text = row[0]  # Text aus der ersten Spalte extrahieren
                    text = classifier._preprocess_text(text)  # Text vorverarbeiten
                    df_list.append({'text': text, 'label': label})
    return pd.DataFrame(df_list)

# ARFF-Konvertierung und Speicherung todo: Weka nimmt SVC/(SVM) arff nicht
def save_as_arff(df, file_name):
    arff_dict = {
        'description': 'News Dataset',
        'relation': 'news',
        'attributes': [('text', 'STRING'), ('label', ['0', '1'])],
        'data': df.values.tolist()
    }

    arff_content = arff.dumps(arff_dict)

    with open(file_name, 'w') as f:
        f.write(arff_content)

# Interpretation für Konfusionsmatrix-Werte
def interpret_and_plot_confusion_matrix(cm, classes,
                                        normalize=False,
                                        title='Confusion matrix',
                                        cmap=plt.cm.Blues):

    # Diese Funktion interpretiert die Konfusionsmatrix und gibt sie als Heatmap aus.

    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    interpretations = {
        'true_positive': f" {cm[0, 0]}",
        'false_positive': f"{cm[0, 1]}",
        'false_negative': f"{cm[1, 0]}",
        'true_negative': f"{cm[1, 1]}"
    }

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.text(0, 0.5, '\n'.join([f"{key}: {value}" for key, value in interpretations.items()]),
             verticalalignment='center', horizontalalignment='left', fontsize=12)
    plt.show()

def plot_metrics(accuracy, precision, recall, f1):
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(10, 5))
    plt.bar(names, values, color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Evaluation Metrics')
    plt.ylim([0, 1])
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    plt.show()

def main():
    # Pfade zu den Fake- und True-News-Datensätzen
    fake_news_path = 'Trainingsdaten/10%/FNT_10'
    true_news_path = 'Trainingsdaten/10%/TNT_10'

    # Laden der Trainingsdaten und Trainieren des Klassifikators
    X_train, X_test, y_train, y_test = load_data(fake_news_path, true_news_path)
    classifier = NewsClassifier()
    classifier.train(X_train, y_train)

    # Training in Schritten
    training_steps = [10, 20, 25, 35, 50, 70, 75, 85, 100]
    train_in_steps(classifier, training_steps, X_test, y_test)

    # Erstellen eines DataFrames mit Trainingsdaten
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})

    # Speichern als ARFF
    save_as_arff(train_df, 'train_data.arff')

    # Evaluierung des Klassifikators
    accuracy, precision, recall, f1, cm = classifier.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print('Confusion Matrix:')
    print(cm)

    # Visualisierung der Evaluierungsmetriken
    plot_metrics(accuracy, precision, recall, f1)

if __name__ == "__main__":
    main()
