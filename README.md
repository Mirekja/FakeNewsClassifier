# Klassifikation von Satire/Fake News

## Modul: Machine Learning

### Autor: Mirco Jablonski


---

## Inhaltsverzeichnis
1. [Überblick](#überblick)
2. [Anforderungen](#anforderungen)
3. [Datenverarbeitung](#datenverarbeitung)
4. [Modelltraining](#modelltraining)
5. [Modellbewertung](#modellbewertung)
6. [Speichern der Daten und Modelle](#speichern-der-daten-und-modelle)
7. [Visualisierung](#visualisierung)
8. [Ausführen der Skripte](#ausführen-der-skripte)

---

## Überblick
Dieses Projekt ist im Rahmen des Bachelormoduls "Machine Learning" im Studiengang Information Science entstanden und zielt darauf ab, Satire/Fake News von echten Nachrichten zu unterscheiden. Dies wird durch die Implementierung eines Textklassifikators erreicht, der mit Machine Learning Algorithmen trainiert wird. Der Code beinhaltet die Vorverarbeitung der Daten, das Training des Modells sowie die Bewertung der Ergebnisse.

## Anforderungen
- Python 3.x
- Os
- Pandas
- NLTK
- Scikit-Learn
- re
- CSV
- numpy

Installieren Sie die benötigten Bibliotheken mit:
```bash
pip install pandas nltk scikit-learn os csv re numpy
```
## Datenverarbeitung

### Textvorverarbeitung
- **Tokenisierung:** Aufteilung des Textes in einzelne Wörter.
- **Normalisierung:** Konvertierung des Textes in Kleinbuchstaben.
- **Lemmatisierung:** Reduktion der Wörter auf ihre Grundform.
- **Entfernung von Stop-Wörtern:** Ausschluss von häufigen Wörtern, die keine Bedeutung tragen.

### Stop-Wörter
Zusätzlich zu den standardmäßigen deutschen Stop-Wörtern werden benutzerdefinierte Stop-Wörter verwendet.

## Modelltraining
Der `NewsClassifier` wird verwendet, um den Textklassifikator zu trainieren. Der Klassifikator verwendet verschiedene Algorithmen (Naive Bayes, SVM, SVC), die auf Wunsch eingestellt werden kann. Der Trainingsprozess erfolgt in frei einstellbaren Schritten, wobei das Modell nach jedem Schritt bewertet wird und die Metriken nach dem Training zeigt.

### Hauptkomponenten:
- **Trainingsmethoden:** `train`, `evaluate`, `classify`
- **Datenvorbereitung:** `load_data`, `load_csv`
- **Speicherung des Modells:** Nach dem Training wird das Model im ARFF- und CSV-Format gespeichert.

## Modellbewertung
Die Leistung des Modells wird mit den Metriken `accuracy`, `precision`, `recall`, `f1_score` und einer Konfusionsmatrix bewertet.

Das `Testing.py`-Skript enthält die Implementierung zur Klassifizierung von Texten und zur Bewertung des Modells. Mittels der erstellten CSV-Datei kann der Klassifikator mit deutschen Nachrichten getestet und evaluiert werden.

## Speichern der Daten und Modelle
- **ARFF-Datei:** Die Trainingsdaten werden in das ARFF-Format konvertiert und gespeichert.
- **CSV:** Die Trainingsdaten werden in das CSV-Format konvertiert und gespeichert.

## Visualisierung
- **Konfusionsmatrix:** Wird als Heatmap visualisiert.
- **Metriken:** `accuracy`, `precision`, `recall`, und `f1_score` werden als Balkendiagramme dargestellt.

## Ausführen der Skripte
Um das Skript auszuführen, stellen Sie sicher, dass die Ordnerstruktur korrekt ist.
Die Ordnerstruktur sollte wie folgt sein:

```plaintext
FakeNewsClassifier
├── Trainingsdaten
│   ├── FNT
│   │   └── Prozentordner
│   │       ├── 10%
│   │       ├── 20%
│   │       └── (...)
│   ├── TNT
│   │   └── Prozentordner
│   │       ├── 10%
│   │       ├── 20%
│   │       └── (...)
```


Führen Sie dann das Skript mit diesem Befehl aus:

```bash
python TK_News.py
```

oder starten Sie das Skript in Ihrer IDE.


Führe das `Testing.py`-Skript mit diesem Befehl aus:

```
python Testing.py
```
oder starten Sie das Skript in Ihrer IDE.

## Hauptfunktion

Die `main`-Funktion koordiniert die zentralen Aufgaben des Programms:

1. **Laden der Trainings- und Testdaten:**
   - Die Daten werden aus den angegebenen Pfaden für Fake- und True-News geladen.
   - Anschließend erfolgt die Aufteilung der Daten in Trainings- und Testsets.


2. **Training des Klassifikators:**
   - Ein neues Objekt der Klasse `NewsClassifier` wird erstellt.
   - Das Modell wird mit den Trainingsdaten mithilfe der `train`-Methode trainiert.


3. **Schrittweises Training**
   - Der Klassifikator wird in mehreren Schritten trainiert.
   - Die Trainingsschritte sind anhand dieser Prozentschritte `10%, 20%, 25%, 35%, 50%, 70%, 75%, 85%, 100%` festgelegt.
   - Die Schritte sind frei wählbar.
   - Die jeweiligen Ordner sollten im Hauptverzeichnis. Passen Sie sonst den Pfad im Code an.


4. **Speichern der Trainingsdaten als ARFF und CSV:**
   - Die Trainingsdaten werden in das ARFF- und CSV-Format konvertiert und gespeichert.


5. **Evaluierung und Visualisierung der Ergebnisse:**
   - Die Leistung des trainierten Modells wird mit den Testdaten bewertet. (Gilt nur für das Training. Für valide Ergebnisse wird die Testing.py verwendet.)
   - Die Evaluierung umfasst die Berechnung von `accuracy`, `precision`, `recall` und `f1_score`.
   - Eine Konfusionsmatrix wird visualisiert, um die Ergebnisse der Klassifikation darzustellen.
   - Metriken wie `accuracy`, `precision`, `recall` und `f1_score` werden als Balkendiagramme visualisiert, um die Leistung des Modells besser zu verstehen.

___
### Quellen:
- [Scikit-Learn Cheat Sheet](https://www.datacamp.com/cheat-sheet/scikit-learn-cheat-sheet-python-machine-learning)
- [Matplotlib Cheat Sheet](https://www.datacamp.com/cheat-sheet/matplotlib-cheat-sheet-plotting-in-python)
- [Fake News Dataset](https://www.kaggle.com/datasets/astoeckl/fake-news-dataset-german)
- [News Dataset](https://www.kaggle.com/datasets/pqbsbk/german-news-dataset)   
