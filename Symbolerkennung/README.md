# Symbolerkennung (UML Component Recognition)

Pipeline zur Erkennung von UML-Diagramm-Komponenten mithilfe eines Vision Transformers (ViT). Es umfasst das Training, die Evaluation und die Inferenz (Erkennung auf neuen Bildern).

## Voraussetzungen

Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

## Ausführreihenfolge

### 1. Training (`train.py`)
Trainiert den Vision Transformer (`google/vit-base-patch16-224`) auf den UML-Daten.
- Lädt Daten aus `data/train` und `data/test`.
- Speichert trainiertes Modell unter `output/weights/best_vit_model.pth`.

```bash
python train.py
```

### 2. Evaluation (`evaluate.py`)
Evaluiert das trainierte Modell auf dem Test-Set und generiert detaillierte Metriken.
- Erstellt einen Klassifikationsbericht in der Konsole.
- Speichert die Ergebnisse als Tabelle in `output/results.md`.

```bash
python evaluate.py
```

### 3. Inferenz (`inference.py`)
Wendet das Modell auf neue, unbekannte Bilder an.
- **Einzelne Symbole:** Bilder in `input/parts/` werden klassifiziert. Ergebnisse landen in `output/metrics_run_*.md`.
- **Ganze Diagramme:** Bilder in `input/diagram/` werden automatisch segmentiert (OpenCV), die Einzelteile klassifiziert und zu einem Mermaid-Graphen zusammengefügt. Die Ergebnisse (Mermaid-Markdown) liegen in `output/`.

```bash
python inference.py
```

## Projektstruktur

- `data/`: Trainings- und Testdaten (nach Klassen sortiert).
- `input/`: Eingabebilder für die Inferenz (`parts` für Symbole, `diagram` für ganze Skizzen).
- `output/`: Speicherort für trainierte Gewichte, Metriken und generierte Mermaid-Diagramme.
- `config.py`: Zentrale Konfiguration (Pfade, Hyperparameter, Klassen).
- `model.py`: Definition des ViT-basierten Klassifizierers.
- `dataset.py`: Datenladen und Vorverarbeitung (Augmentierung).
- `graph_reconstruction.py`: Logik zur Wiederherstellung der Diagrammstruktur aus erkannten Symbolen.
