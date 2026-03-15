import os
import cv2
import glob
import torch
from typing import List
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr

from ocr_utils import (detect_text_regions, sort_regions_reading_order,
                        split_wide_crop_into_word_chunks, recognize_handwriting,
                        remove_grid_background)
from text_utils import (correct_text, dictionary_score, best_text,
                         bert_correct_sentence, lm_sentence_score)

# --- Modelle laden ---
# GPU verwenden, falls verfügbar, sonst CPU-Fallback
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"System: {device}")

# Deutsches TrOCR-Modell für handgeschriebene Texte (finetuned auf deutsche Handschrift)
model_name = "fhswf/TrOCR_german_handwritten"
print(f"Lade TrOCR Modell ({model_name})...")
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)

# EasyOCR als zweiter OCR-Kanal (Deutsch), liefert Bounding-Boxes + Konfidenz
print("Lade EasyOCR...")
reader = easyocr.Reader(['de'])
print("Modelle bereit!\n")


def recognize(crop, max_new_tokens=128):
    """Wrapper für recognize_handwriting mit vorgeladenem Prozessor und Modell."""
    return recognize_handwriting(crop, processor, model, device, dictionary_score, max_new_tokens)


def get_handwriting_image_paths() -> List[str]:
    """Sucht alle Handschrift-Bilder (jpg/jpeg/png) im sketch-data-Verzeichnis.

    Durchsucht sowohl das direkte Unterverzeichnis als auch alle verschachtelten
    Ordner (rekursiv) nach dem Muster sketch-data/**/Handschrift/.

    Returns:
        Sortierte Liste eindeutiger Bildpfade.
    """
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join("sketch-data", "Handschrift", ext)))
        paths.extend(glob.glob(os.path.join("sketch-data", "**", "Handschrift", ext), recursive=True))
    return sorted(set(paths))


# --- Hauptschleife ---
print("=== Ordner: Handschrift ===")
images = get_handwriting_image_paths()
if not images:
    print("Keine Bilder gefunden. Erwartet: sketch-data/**/Handschrift/")

for img_path in images:
    print(f"\nBild: {os.path.basename(img_path)}")
    img = cv2.imread(img_path)
    if img is None:
        print("  -> Bild konnte nicht geladen werden.")
        continue

    # Gitterhintergrund (Karopapier) entfernen – verbessert EasyOCR + TrOCR.
    img = remove_grid_background(img)

    # EasyOCR erkennt Textregionen als Bounding-Boxes mit Rohtext und Konfidenz
    results, region_mode = detect_text_regions(img_path, reader)
    if not results:
        print("  -> Kein Text gefunden.")
        continue

    print(f"  -> {len(results)} Textbereiche gefunden (Modus: {region_mode}):")
    results = sort_regions_reading_order(results)
    final_region_texts: List[str] = []  # Ergebnisse aller Regionen für den Gesamtsatz

    for i, (bbox, init_text, prob) in enumerate(results):
        # Bounding-Box mit Padding ausschneiden (12 % der Regionsgröße, mind. 4 px)
        x_coords = [int(p[0]) for p in bbox]
        y_coords = [int(p[1]) for p in bbox]
        pad = max(4, int(0.12 * max(max(x_coords)-min(x_coords), max(y_coords)-min(y_coords))))
        x1 = max(0, min(x_coords) - pad);  x2 = min(img.shape[1], max(x_coords) + pad)
        y1 = max(0, min(y_coords) - pad);  y2 = min(img.shape[0], max(y_coords) + pad)
        crop = img[y1:y2, x1:x2]

        # Breite Regionen werden in Wort-Chunks aufgeteilt, damit TrOCR besser erkennt
        chunks = split_wide_crop_into_word_chunks(crop)
        trocr_text = " ".join(filter(None, [recognize(c) for c in chunks])) if len(chunks) > 1 else recognize(crop)

        # Beide OCR-Ausgaben wörterbuchbasiert korrigieren
        corrected_easyocr = correct_text(init_text)
        corrected_trocr   = correct_text(trocr_text)
        # Bestes Ergebnis für diese Region auswählen
        final_text, source = best_text(corrected_easyocr, prob, corrected_trocr)
        final_region_texts.append(final_text)

        print(f"    Region {i+1}:")
        print(f"      Roh/EasyOCR        : '{init_text}' (conf={prob:.2f})")
        print(f"      EasyOCR + Korrektur: '{corrected_easyocr}'")
        print(f"      TrOCR (roh)        : '{trocr_text}'")
        print(f"      TrOCR + Korrektur  : '{corrected_trocr}'")
        print(f"      >>> Finales Ergebnis: '{final_text}'  [{source}]")

    # Alle Regionen zum Gesamtsatz zusammenfügen
    region_sentence = " ".join(filter(None, final_region_texts)).strip()
    # BERT korrigiert noch verbleibende nicht-deutsche Wörter im Kontext des Gesamtsatzes
    bert_sentence_raw = bert_correct_sentence(region_sentence)

    # BERT-Korrektur nur akzeptieren, wenn sie objektiv besser ist:
    # - Wörterbuch-Score darf nicht schlechter werden
    # - LM-Score muss sich um mindestens 0.2 verbessern
    # - Satzlänge darf sich um maximal 1 Wort verändern
    base_dict = dictionary_score(region_sentence)
    cand_dict = dictionary_score(bert_sentence_raw)
    base_lm = lm_sentence_score(region_sentence)
    cand_lm = lm_sentence_score(bert_sentence_raw)
    similar_len = abs(len(bert_sentence_raw.split()) - len(region_sentence.split())) <= 1
    improved = (cand_dict >= base_dict and cand_lm > base_lm + 0.2 and similar_len)
    bert_sentence = bert_sentence_raw if improved else region_sentence

    print(f"  >>> Gesamt (Regionen)  : '{region_sentence}'")
    print(f"  >>> BERT-Korrektur:      '{bert_sentence}'")

print("\nAlle Tests abgeschlossen.")
