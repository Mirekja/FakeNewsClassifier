# Klassifikation von Satire/Fake News

## Modul: Machine Learning bei Andreas Heß

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
8. [Ausführen des Skripts](#ausführen-des-skripts)

---

## Überblick
Dieses Projekt ist im Rahmen des Bachelormoduls "Machine Learning" im Studiengang Information Science entstanden und zielt darauf ab, Satire/Fake News von echten Nachrichten zu unterscheiden. Dies wird durch die Implementierung eines Textklassifikators erreicht, der mit verschiedenen Machine Learning Algorithmen trainiert wird. Der Code beinhaltet die Vorverarbeitung der Daten, das Training des Modells, die Bewertung der Ergebnisse und die Visualisierung der Leistung.

## Anforderungen
- Python 3.x
- Pandas
- NLTK
- Scikit-Learn
- Joblib
- Matplotlib
- Numpy

Installieren Sie die benötigten Bibliotheken mit:
```bash
pip install pandas nltk scikit-learn joblib matplotlib numpy
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
Der `NewsClassifier` wird verwendet, um den Textklassifikator zu trainieren. Der Klassifikator verwendet einen SVM-Algorithmus (Support Vector Machine) mit einem linearen Kernel. Der Trainingsprozess erfolgt in mehreren Schritten, wobei das Modell nach jedem Schritt gespeichert und bewertet wird.

### Hauptkomponenten:
- **Trainingsmethoden:** `train`, `evaluate`, `classify`
- **Datenvorbereitung:** `load_data`, `load_csv`
- **Speicherung des Modells:** Nach jedem Trainingsschritt wird das Modell mit `joblib` gespeichert.

## Modellbewertung
Die Leistung des Modells wird mit den Metriken `accuracy`, `precision`, `recall`, `f1_score` und einer Konfusionsmatrix bewertet.

## Speichern der Daten und Modelle
- **ARFF-Datei:** Die Trainingsdaten werden in das ARFF-Format konvertiert und gespeichert.
- **Joblib:** Der trainierte Klassifikator wird nach jedem Trainingsschritt gespeichert.

## Visualisierung
- **Konfusionsmatrix:** Wird als Heatmap visualisiert.
- **Metriken:** `accuracy`, `precision`, `recall`, und `f1_score` werden als Balkendiagramme dargestellt.

## Ausführen des Skripts
Um das Skript auszuführen, stellen Sie sicher, dass die Ordnerstruktur korrekt ist.
Die Ordnerstruktur sollte wie folgt sein:

```plaintext
FakeNewsClassifier
├── Trainingsdaten
    ├── Prozentordner
        ├── FNT_{%}
        └── TNT_{%}
```


Führen Sie dann das Skript mit:

```bash
python TK_News.py
```

## Hauptfunktion

Die `main`-Funktion koordiniert die zentralen Aufgaben des Programms:

1. **Laden der Trainings- und Testdaten:**
   - Die Daten werden aus den angegebenen Pfaden für Fake- und True-News geladen.
   - Anschließend erfolgt die Aufteilung der Daten in Trainings- und Testsets.

2. **Training des Klassifikators:**
   - Ein neues Objekt der Klasse `NewsClassifier` wird erstellt.
   - Das Modell wird mit den Trainingsdaten mithilfe der `train`-Methode trainiert.

3. **Schrittweises Training mit Zwischenspeicherung:**
   - Der Klassifikator wird in mehreren Schritten trainiert, wobei das Modell nach jedem Schritt gespeichert wird.
   - Die Trainingsfortschritte werden in Schritten von 10% bis 100% überprüft und bewertet.

4. **Speichern der Trainingsdaten als ARFF:**
   - Die Trainingsdaten werden in das ARFF-Format konvertiert und in einer Datei gespeichert.

5. **Evaluierung und Visualisierung der Ergebnisse:**
   - Die Leistung des trainierten Modells wird mit den Testdaten bewertet.
   - Die Evaluierung umfasst die Berechnung von `accuracy`, `precision`, `recall` und `f1_score`.
   - Eine Konfusionsmatrix wird visualisiert, um die Ergebnisse der Klassifikation darzustellen.
   - Metriken wie `accuracy`, `precision`, `recall` und `f1_score` werden als Balkendiagramme visualisiert, um die Leistung des Modells besser zu verstehen.

Die `main`-Funktion ist der zentrale Einstiegspunkt des Skripts und koordiniert alle Schritte vom Laden der Daten bis zur Auswertung der Klassifikationsergebnisse.
___
### Quellen:
- [Scikit-Learn Cheat Sheet](https://www.datacamp.com/cheat-sheet/scikit-learn-cheat-sheet-python-machine-learning)
- [Matplotlib Cheat Sheet](https://www.datacamp.com/cheat-sheet/matplotlib-cheat-sheet-plotting-in-python)

