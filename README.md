## Projektstruktur
```
NLP-Projekt-WS2526-Abgabe/
│
├── Anfangsidee/                  # Erste Projektidee und initiale Prototypen (Archiv)
│   ├── output/                   # Generierte Ausgaben der Anfangsidee
│   │   └── flow-chart_flowchart (1).mmd  # Mermaid-Flowchart-Diagramm
│   ├── sketch-data/              # Skizzen und Rohdaten der ersten Idee
│   ├── project.ipynb             # Jupyter Notebook des ersten Ansatzes
│   ├── run_pipeline.py           # Skript zum Ausführen der Pipeline
│   └── README.md                 # Dokumentation zur Anfangsidee
│
├── Ausarbeitung/                 # Schriftliche Ausarbeitung des Projekts
│   └── Ausarbeitung_ML_Diagramm_Handschrifterkennung.pdf
│
├── Handschrifterkennung/         # Hauptmodul zur Erkennung handgeschriebener Texte
│   ├── output/                   # Ausgaben der OCR-Pipeline
│   ├── sketch-data/              # Eingabebilder und Testdaten
│   ├── ocr_utils.py              # Hilfsfunktionen für die OCR-Verarbeitung
│   ├── run_pipeline.py           # Einstiegspunkt zum Starten der Pipeline
│   ├── text_utils.py             # Hilfsfunktionen für Textverarbeitung
│   ├── train_trocr.py            # Skript zum Training des TrOCR-Modells
│   └── README.md                 # Dokumentation zur Handschrifterkennung
│
├── Symbolerkennung/              # Modul zur Erkennung von UML-Diagramm-Symbolen
│   ├── data/                     # Trainingsdaten für das Modell
│   ├── input/                    # Eingabebilder für die Inferenz
│   ├── output/                   # Ergebnisse der Symbolerkennung
│   ├── raw sources/              # Rohdatenquellen
│   ├── config.py                 # Konfigurationsparameter
│   ├── dataset.py                # Datensatz-Klasse und Vorverarbeitung
│   ├── evaluate.py               # Evaluierung des Modells
│   ├── graph_reconstruction.py   # Rekonstruktion des Graphen aus erkannten Symbolen
│   ├── inference.py              # Inferenz auf neuen Bildern
│   ├── model.py                  # Modelldefinition (Vision Transformer)
│   ├── train.py                  # Training des Modells
│   ├── requirements.txt          # Abhängigkeiten
│   └── README.md                 # Dokumentation zur Symbolerkennung
│
├── .gitignore
└── README.md                     # Diese Datei
```
