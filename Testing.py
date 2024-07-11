import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def train_model(csv_file):
    try:
        data = pd.read_csv(csv_file, sep=',')
        data = data.dropna()  # Entfernen von Zeilen mit NaN-Werten
        X = data['text'].astype(str).tolist()  # Liste der Texte
        y = data['label'].astype(str).tolist()  # Liste der Labels
    except IOError:
        print(f"Fehler: Die Datei '{csv_file}' konnte nicht geöffnet werden.")
        return None, None, None
    except KeyError as e:
        print(f"Fehler: Spalte '{e}' wurde in der CSV-Datei nicht gefunden.")
        return None, None, None

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
        #('classifier', svm.SVC(kernel='linear'))
    ])

    # Trainieren
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    return pipeline, np.unique(y)

def classify_text(model, class_values, text):
    prediction = model.predict([text])
    predicted_index = int(prediction[0])
    predicted_class = class_values[predicted_index]
    return predicted_class

if __name__ == '__main__':
    csv_file = 'Classifier/Naive Bayes 50.csv'
    model, class_values = train_model(csv_file)

    if model is not None:
        input_text = input("Geben Sie den Text ein, den Sie klassifizieren möchten: ")
        predicted_class = classify_text(model, class_values, input_text)
        print(f"Der eingegebene Text wird als '{predicted_class}' klassifiziert.") # 1 ist wahr, 0 ist falsch