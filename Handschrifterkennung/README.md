# Handschrifterkennung – Dokumentation

Dieses Projekt erkennt handgeschriebenen deutschen Text auf Bildern mithilfe einer mehrstufigen OCR-Pipeline (TrOCR + EasyOCR + BERT-Korrektur).

---

## Voraussetzungen

- Python 3.9+
- CUDA-fähige GPU (empfohlen, CPU funktioniert aber auch)
- Benötigte Pakete (z. B. via `pip install`):
  - `transformers`
  - `easyocr`
  - `torch`
  - `opencv-python`
  - `Pillow`
  - `pyspellchecker`
  - `python-Levenshtein`

---

## Projektstruktur

```
Handschrifterkennung/
├── run_pipeline.py          # Hauptskript – hier wird die Pipeline gestartet
├── ocr_utils.py             # Bildvorverarbeitung, Texterkennung, Regionerkennung
├── text_utils.py            # Rechtschreibkorrektur, BERT-Korrektur, Scoring
├── train_trocr.py           # Training-Skript (archiviert, nicht aktiv genutzt)
├── models/                  # Gespeicherte Modelle
│   ├── trocr-finetuned-de/
│   └── trocr-finetuned-handwritten/
├── sketch-data/
│   └── Handschrift/         # Hier kommen die Testbilder rein
└── output/                  # Hier landen die Ergebnisse
```

---

## Testen – Schritt für Schritt

### 1. Bild vorbereiten

Schreibe einen Satz (oder mehrere) auf ein Blatt Papier und fotografiere oder scanne es.
Das Bild muss im Format **JPG oder PNG** vorliegen.

> Tipps für beste Ergebnisse:
> - Gute Beleuchtung, kein Schatten über dem Text
> - Dunkle Tinte auf hellem Papier
> - Text möglichst horizontal ausrichten
> - Nicht zu klein schreiben

### 2. Bild in den richtigen Ordner legen

Lege das Bild in folgenden Ordner:

```
sketch-data/Handschrift/
```

Beispiel:
```
sketch-data/Handschrift/mein_satz.jpg
```

### 3. Pipeline starten

Führe das Hauptskript aus:

```bash
python run_pipeline.py
```

Das Skript verarbeitet automatisch alle Bilder im Ordner `sketch-data/Handschrift/` und gibt den erkannten Text in der Konsole aus.

### 4. Ergebnis lesen

Die Ausgabe erscheint in der Konsole, z. B.:

```
[mein_satz.jpg] → "Der Hund läuft durch den Park."
```

---

## Testsätze

Hier sind geeignete Sätze zum Ausprobieren – von einfach bis anspruchsvoll:

### Einfach
- Der Hund läuft durch den Park.
- Heute ist ein schöner Tag.
- Ich trinke gerne Kaffee am Morgen.
- Das Buch liegt auf dem Tisch.
- Mein Name ist Anna.
- Wir fahren nächste Woche in den Urlaub.
- Die Sonne scheint hell und warm.
- Er kauft Brot und Milch im Laden.
- Das Kind spielt im Garten.
- Sie liest jeden Abend ein Buch.

### Mit Umlauten & Satzzeichen
- Über den großen Fluss fährt täglich ein Zug.
- Könntest du mir bitte helfen? Ich wäre sehr dankbar!
- Möchtest du Kaffee oder Tee trinken?

### Pangramme (enthalten fast alle Buchstaben – ideal zum Testen)
- Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich.
- Fix, Schwyz! quäkt Jürgen blöd vom Paß.

---

## Wie die Pipeline funktioniert

```
Eingabebild (JPG/PNG)
    ↓
[1] Gitterhintergrund entfernen (Liniertes Papier)
    ↓
[2] Textregionen erkennen (EasyOCR)
    ↓
[3] Regionen nach Lesereihenfolge sortieren
    ↓
[4] Für jede Region:
    ├─ Bild zuschneiden & auffüllen
    ├─ Breite Bereiche in Wörter aufteilen
    ├─ TrOCR: 3 Bildvarianten → bestes Ergebnis wählen
    ├─ Rechtschreibkorrektur (EasyOCR + TrOCR)
    └─ Besten Text auswählen
    ↓
[5] Alle Regionen zusammenführen
    ↓
[6] BERT-Satzkorrektur (kontextbasiert)
    ↓
Erkannter Text
```

### Kurzübersicht der Schritte

| Schritt | Was passiert |
|--------|--------------|
| Vorverarbeitung | Gitterlinien werden entfernt, Kontrast verbessert |
| Regionerkennung | EasyOCR findet Textbereiche im Bild |
| Texterkennung | TrOCR liest jeden Bereich in 3 Varianten (Original, CLAHE, hochskaliert) |
| Rechtschreibung | Wörter werden gegen deutsches Wörterbuch geprüft |
| BERT-Korrektur | Kontext-aware Nachkorrektur mit `dbmdz/bert-base-german-cased` |

---

## Verwendete Modelle

| Modell | Zweck |
|--------|-------|
| `fhswf/TrOCR_german_handwritten` | Hauptmodell für Handschrifterkennung |
| `dbmdz/bert-base-german-cased` | Kontextbasierte Satzkorrektur |
| EasyOCR (Deutsch) | Textregionenerkennung & Fallback-OCR |

---

## Häufige Probleme

| Problem | Lösung |
|--------|--------|
| Text wird nicht erkannt | Bild heller/kontrastreicher machen, größer schreiben |
| Falsche Buchstaben | Deutlicher und weniger verschnörkelt schreiben |
| Langsame Verarbeitung | GPU verwenden (CUDA), Bildgröße reduzieren |
| Bild wird nicht gefunden | Sicherstellen, dass das Bild in `sketch-data/Handschrift/` liegt |

---

## Training (optional)

Das Skript `train_trocr.py` ermöglicht das Feintuning eines eigenen TrOCR-Modells.
Es wird **nicht** genutzt, da das vortrainierte Modell besser abschneidet.

Format der Trainingsdaten (CSV):
```
image_path,text
bilder/satz1.jpg,Das ist ein Beispielsatz.
bilder/satz2.jpg,Noch ein Satz zum Trainieren.
```

Start:
```bash
python train_trocr.py --data meine_daten.csv
```
