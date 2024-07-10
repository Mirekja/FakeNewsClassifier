import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.pipeline import Pipeline
import numpy as np

def train_model(csv_file):
    try:
        data = pd.read_csv(csv_file, sep=',')
        data = data.dropna()  # Entfernen von Zeilen mit NaN-Werten
        X = data['text'].astype(str).tolist()  # Liste der Texte
        y = data['label'].astype(str).tolist()  # Liste der Labels
    except IOError:
        print(f"Fehler: Die Datei '{csv_file}' konnte nicht geöffnet werden.")
        return None, None
    except KeyError as e:
        print(f"Fehler: Spalte '{e}' wurde in der CSV-Datei nicht gefunden.")
        return None, None

    # Pipeline für Naive Bayes
    pipeline_nb = Pipeline([
        ('vectorizer', CountVectorizer()),
        #('vectorizer', TfidfVectorizer()),
        #('classifier', MultinomialNB())
        ('classifier', svm.SVC(kernel='linear'))
    ])

    # Trainieren Modells
    pipeline_nb.fit(X, y)

    return pipeline_nb, np.unique(y)

def classify_text(model, class_values, text):
    prediction = model.predict([text])
    predicted_index = int(prediction[0])
    predicted_class = class_values[predicted_index]
    return predicted_class



if __name__ == '__main__':
    csv_file = 'Classifier/SVM 50.csv'
    model_nb, class_values_nb = train_model(csv_file)

    if model_nb is not None:
        input_text = input("Geben Sie den Text ein, den Sie klassifizieren möchten: ")
        predicted_class_nb = classify_text(model_nb, class_values_nb, input_text)
        print(f"Der eingegebene Text wird als '{predicted_class_nb}' klassifiziert.") # 1 ist wahr, 0 ist falsch