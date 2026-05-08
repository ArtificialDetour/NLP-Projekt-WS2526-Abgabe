# NLP-Projekt WS25/26 – UML-Diagramm-Erkennung aus Handskizzen

Dieses Projekt erkennt handgezeichnete UML-Aktivitätsdiagramme. Es besteht aus zwei unabhängigen Modulen:

| Modul | Aufgabe |
|---|---|
| **Symbolerkennung** | Erkennt UML-Symbole (Start, Ende, Zustand, Entscheidung, Aktion) per Vision Transformer |
| **Handschrifterkennung** | Liest handgeschriebenen Text innerhalb der Symbole per TrOCR + BERT |

Die schriftliche Ausarbeitung liegt unter [Ausarbeitung/Ausarbeitung_ML_Diagramm_Handschrifterkennung.pdf](Ausarbeitung/Ausarbeitung_ML_Diagramm_Handschrifterkennung.pdf).

---

## Voraussetzungen

- Python 3.9 oder neuer
- GPU mit CUDA (empfohlen, CPU funktioniert aber auch)
- `pip` zum Installieren der Abhängigkeiten

---

## Schnellstart

### 1. Abhängigkeiten installieren

Beide Module haben eigene Pakete. Alles auf einmal installieren:

```bash
pip install torch torchvision transformers scikit-learn matplotlib seaborn networkx \
            easyocr opencv-python Pillow pyspellchecker python-Levenshtein
```

Oder nur für die Symbolerkennung:

```bash
pip install -r Symbolerkennung/requirements.txt
```

---

## Modul 1: Symbolerkennung

Erkennt UML-Symbole (action, choice, ending, start, state) auf handgezeichneten Diagrammbildern und rekonstruiert daraus einen Mermaid-Graphen.

### Schritt 1 – Modell trainieren

Trainingsdaten liegen in `Symbolerkennung/data/train/` und `Symbolerkennung/data/test/` (nach Klassen sortierte Ordner).

```bash
cd Symbolerkennung
python train.py
```

Das trainierte Modell wird unter `output/weights/best_vit_model.pth` gespeichert.

### Schritt 2 – Modell evaluieren (optional)

```bash
python evaluate.py
```

Gibt einen Klassifikationsbericht aus und speichert die Ergebnisse in `output/results.md`.

### Schritt 3 – Inferenz auf neuen Bildern

**Einzelne Symbole klassifizieren:**

Bilder (JPG/PNG) in `Symbolerkennung/input/parts/` legen, dann:

```bash
python inference.py
```

Ergebnisse erscheinen in `output/metrics_run_*.md`.

**Ganze Diagramme verarbeiten:**

Bilder in `Symbolerkennung/input/diagram/` legen. Das Skript segmentiert das Bild automatisch, klassifiziert die Symbole und erstellt einen Mermaid-Graphen in `output/`.

```bash
python inference.py
```

### Konfiguration

Alle Pfade und Hyperparameter sind zentral in [Symbolerkennung/config.py](Symbolerkennung/config.py) einstellbar.

---

## Modul 2: Handschrifterkennung

Liest handgeschriebenen deutschen Text auf Fotos oder Scans mithilfe einer mehrstufigen OCR-Pipeline (TrOCR + EasyOCR + BERT-Nachkorrektur).

### Schritt 1 – Bild vorbereiten

Bild (JPG oder PNG) in folgenden Ordner legen:

```
Handschrifterkennung/sketch-data/Handschrift/
```

Tipps für beste Ergebnisse:
- Gute Beleuchtung, kein Schatten
- Dunkle Tinte auf hellem Papier
- Text möglichst horizontal ausrichten

### Schritt 2 – Pipeline starten

```bash
cd Handschrifterkennung
python run_pipeline.py
```

Das Skript verarbeitet alle Bilder im Ordner automatisch und gibt den erkannten Text in der Konsole aus:

```
[mein_satz.jpg] → "Der Hund läuft durch den Park."
```

### Verwendete Modelle

| Modell | Zweck |
|---|---|
| `fhswf/TrOCR_german_handwritten` | Hauptmodell Handschrifterkennung |
| `dbmdz/bert-base-german-cased` | Kontextbasierte Satzkorrektur |
| EasyOCR (Deutsch) | Textregionenerkennung & Fallback |

Die Modelle werden beim ersten Start automatisch von Hugging Face heruntergeladen.

### Häufige Probleme

| Problem | Lösung |
|---|---|
| Text wird nicht erkannt | Bild heller/kontrastreicher machen |
| Falsche Buchstaben | Deutlicher schreiben, weniger Schnörkel |
| Langsame Verarbeitung | GPU (CUDA) verwenden oder Bildgröße reduzieren |
| Bild nicht gefunden | Pfad prüfen: `sketch-data/Handschrift/dateiname.jpg` |

---

## Projektstruktur

```
NLP-Projekt-WS2526-Abgabe/
│
├── Symbolerkennung/              # UML-Symbol-Klassifikation (Vision Transformer)
│   ├── data/                     # Trainingsdaten (nach Klassen sortiert)
│   ├── input/
│   │   ├── parts/                # Einzelne Symbole für Inferenz
│   │   └── diagram/              # Ganze Diagrammbilder für Inferenz
│   ├── output/                   # Modelgewichte, Metriken, Mermaid-Ausgaben
│   ├── config.py                 # Zentrale Konfiguration
│   ├── model.py                  # ViT-Klassifizierer
│   ├── dataset.py                # Datenladen & Augmentierung
│   ├── train.py                  # Training
│   ├── evaluate.py               # Evaluation
│   ├── inference.py              # Inferenz auf neuen Bildern
│   ├── graph_reconstruction.py   # Mermaid-Graph-Rekonstruktion
│   └── requirements.txt
│
├── Handschrifterkennung/         # OCR-Pipeline für handgeschriebenen Text
│   ├── sketch-data/Handschrift/  # Eingabebilder hier ablegen
│   ├── output/                   # Erkannte Texte
│   ├── run_pipeline.py           # Pipeline starten
│   ├── ocr_utils.py              # Bildvorverarbeitung & Texterkennung
│   └── text_utils.py             # Rechtschreib- & BERT-Korrektur
│
├── Ausarbeitung/                 # Schriftliche Projektausarbeitung (PDF)
├── Anfangsidee/                  # Archiv: erster Prototyp (nicht produktiv)
└── README.md
```
