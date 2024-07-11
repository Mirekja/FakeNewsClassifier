import pandas as pd
import os
import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# NLTK-Daten herunterladen
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Zusätzliche Stop-Wörter
custom_stop_words = ['postillon', 'anzeige', 'anzeig', 'titletext']

class NewsClassifier:
    def __init__(self, classifier_type='svm'):
        # Kombinierte Liste von Stop-Wörtern
        preprocessed_stop_words = [self._preprocess_word(word) for word in custom_stop_words]
        stop_words = stopwords.words('german') + preprocessed_stop_words
        self.vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1, 3))

        # Initialisierung des Klassifikators
        if classifier_type == 'naive_bayes':
            self.clf = MultinomialNB()
        elif classifier_type == 'svm':
            self.clf = svm.SVC(kernel='linear')
        elif classifier_type == 'linear_svc':
            self.clf = LinearSVC()
        else:
            raise ValueError("Unsupported classifier type")

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
        fake_news_path = f'Trainingsdaten/FN_Data/{step}%'
        true_news_path = f'Trainingsdaten/TN_Data/{step}%'
        X_train_new, _, y_train_new, _ = load_data(fake_news_path, true_news_path)

        if not hasattr(classifier, 'X_train') or not hasattr(classifier, 'y_train'):
            classifier.X_train, classifier.y_train = X_train_new, y_train_new
        else:
            classifier.X_train = pd.concat([classifier.X_train, X_train_new])
            classifier.y_train = pd.concat([classifier.y_train, y_train_new])

        classifier.train(classifier.X_train, classifier.y_train)


        # Auswertung des Klassifikators nach jedem Schritt
        accuracy, precision, recall, f1, cm = classifier.evaluate(X_test, y_test)
        print(f'Accuracy after {step}% training: {accuracy:.4f}')
        print(f'Precision after {step}% training: {precision:.4f}')
        print(f'Recall after {step}% training: {recall:.4f}')
        print(f'F1 Score after {step}% training: {f1:.4f}')
        print('Confusion Matrix after update:')
        print(cm)

        # Ausgabe der Anzahl der Trainings- und Testdatensätze
        print(f'Trainingsdatensätze: {len(classifier.X_train)}')
        print(f'Testdatensätze: {len(X_test)}')


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
        for encoding in ['utf-8', 'iso-8859-1', 'latin1']:
            try:
                with open(file, 'r', encoding=encoding) as csv_file:
                    content = csv_file.read().replace('\0', '')  # Nullzeichen entfernen
                    reader = csv.reader(content.splitlines())
                    for row in reader:
                        if len(row) > 0:  # Überprüfe, ob die Zeile nicht leer ist
                            text = row[0]  # Text aus der ersten Spalte extrahieren
                            text = classifier._preprocess_text(text)  # Text vorverarbeiten
                            df_list.append({'text': text, 'label': label})
                break  # Wenn das Lesen erfolgreich war, brechen wir die Schleife ab
            except (UnicodeDecodeError, csv.Error):
                continue  # Wenn ein Fehler auftritt, probieren wir das nächste Encoding

    return pd.DataFrame(df_list)


# ARFF-Konvertierung
def save_as_arff(df, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write('@relation news\n\n')
        f.write('@attribute text string\n')
        f.write('@attribute label {0, 1}\n')
        f.write('@data\n')

        for index, row in df.iterrows():
            text = row['text'].replace(',', '\\,') if pd.notnull(row['text']) else ''  # Kommas im Text ersetzen
            label = row['label']

            f.write(f"'{text}', {label}\n")

# CSV-Konvertierung
def save_as_csv(df, file_name):
    df.to_csv(file_name, index=False, encoding='utf-8')


def main():
    # Benutzer nach dem gewünschten Klassifikator fragen
    classifier_type = input("Wählen Sie den Klassifikator (naive_bayes, svm, linear_svc): ").strip().lower()
    if classifier_type not in ['naive_bayes', 'svm', 'linear_svc']:
        print("Ungültige Auswahl. Bitte wählen Sie zwischen 'naive_bayes', 'svm' oder 'linear_svc'.")
        return
    print("Sie haben",classifier_type,"ausgewählt")

    # Pfade zu den Fake- und True-News-Datensätzen
    fake_news_path = 'Trainingsdaten/FN_Data/10%'
    true_news_path = 'Trainingsdaten/TN_Data/10%'

    # Laden der Trainingsdaten und Trainieren des Klassifikators
    X_train, X_test, y_train, y_test = load_data(fake_news_path, true_news_path)
    classifier = NewsClassifier(classifier_type=classifier_type)
    classifier.train(X_train, y_train)

    # Training in Schritten
    training_steps = [50]
    train_in_steps(classifier, training_steps, X_test, y_test)

    # Speichern der Trainingsdaten als ARFF und CSV
    train_df = pd.DataFrame({'text': classifier.X_train, 'label': classifier.y_train})
    save_as_arff(train_df, 'Classifier/train_data.arff')
    save_as_csv(train_df, 'Classifier/train_data.csv')

    # Evaluierung des Klassifikators
    accuracy, precision, recall, f1, cm = classifier.evaluate(X_test, y_test)
    print(f'Final Accuracy: {accuracy}')
    print(f'Final Precision: {precision}')
    print(f'Final Recall: {recall}')
    print(f'Final F1 Score: {f1}')
    print('Confusion Matrix:')
    print(cm)


if __name__ == "__main__":
    main()
