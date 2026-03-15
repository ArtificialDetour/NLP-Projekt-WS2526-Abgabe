import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple


def remove_grid_background(img: np.ndarray) -> np.ndarray:
    """Entfernt Gitternetz-/Linienpapier-Hintergrund per morphologischer Normalisierung.

    Funktioniert so: Ein grosser Dilate-Kernel schaetzt den hellen Hintergrund
    (Text ist dunkel und wird 'weggedilated'). Division durch diesen Hintergrund
    macht die Farbe gleichmaessig weiss und hebt Text staerker hervor.

    Args:
        img: BGR-Bild als NumPy-Array.

    Returns:
        Normalisiertes BGR-Bild mit reduziertem Gitterhintergrund.
    """
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Kernel gross genug, um Gitterabstand zu ueberbruecken, aber Buchstaben nicht.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    # Hintergrundschätzung: Dilation löscht dunkle Textpixel, behält hellen Hintergrund.
    background = cv2.dilate(gray, kernel)
    # Divide-Normalisierung: hellt Hintergrund auf, laesst Text dunkel.
    normalized = cv2.divide(gray.astype(np.float32), background.astype(np.float32), scale=255.0)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def preprocess_variants(image_crop: np.ndarray) -> List[np.ndarray]:
    """Erzeugt drei Bildvarianten für robustere TrOCR-Erkennung.

    Varianten:
    1. Original mit weißem Rand (24 px Padding)
    2. CLAHE-Kontrastverbesserung (clipLimit=2.2)
    3. CLAHE-Bild 2x hochskaliert (mindestens 64×64 px)

    Alle Varianten werden zuerst gitter-normalisiert.

    Args:
        image_crop: Ausgeschnittener Bildbereich (BGR).

    Returns:
        Liste mit bis zu drei vorverarbeiteten BGR-Bildern.
        Leere Liste, wenn der Crop ungültig ist.
    """
    if image_crop is None or image_crop.size == 0:
        return []
    # Gitter entfernen bevor weitere Varianten erzeugt werden.
    image_crop = remove_grid_background(image_crop)

    # Variante 1: weißer Rand für bessere TrOCR-Eingabe (Modell erwartet Padding)
    base = cv2.copyMakeBorder(image_crop, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    # Variante 2: CLAHE-Kontrastverstärkung (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrasted = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)

    # Variante 3: 2x Hochskalierung für kleine Crops (verbessert Detailerkennung)
    upscaled = cv2.resize(
        contrasted,
        (max(64, contrasted.shape[1] * 2), max(64, contrasted.shape[0] * 2)),
        interpolation=cv2.INTER_CUBIC,
    )

    return [base, contrasted, upscaled]


def split_wide_crop_into_word_chunks(crop: np.ndarray) -> List[np.ndarray]:
    """Teilt sehr breite Bounding-Boxen (Seitenverhältnis > 5.0) in Wort-Chunks auf.

    TrOCR erkennt einzelne Wörter zuverlässiger als ganze Zeilen. Daher werden
    breite Crops anhand von vertikalen Leerraumspalten gesplittet.

    Ablauf:
    1. Binäres Invertbild (Tinte = weiß) erzeugen
    2. Vertikales Intensitätsprofil (Spaltensummen) berechnen
    3. Spalten mit < 10 % des Maximums als Lücken markieren
    4. Lücken ab Mindestbreite als Trennpunkte verwenden

    Falls mehr als 6 Chunks entstehen (zu fragmentiert), wird der Original-Crop
    zurückgegeben.

    Args:
        crop: BGR-Bildausschnitt.

    Returns:
        Liste von Teilbildern (Chunks) oder [crop] wenn kein Split sinnvoll ist.
    """
    if crop is None or crop.size == 0:
        return []
    h, w = crop.shape[:2]
    # Nur wirklich breite Crops aufteilen (Seitenverhältnis > 5)
    if h < 12 or w < 24 or (w / max(1, h)) < 5.0:
        return [crop]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptives Schwellwertverfahren: Tinte wird weiß (invertiert) für Profilberechnung
    bin_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7)

    # Spalten-Intensitätsprofil: Summe der Tintenpixel je Spalte, leicht geglättet
    profile = cv2.GaussianBlur(bin_inv.sum(axis=0).astype(np.float32).reshape(1, -1), (1, 0), sigmaX=2).reshape(-1)
    if profile.max() <= 0:
        return [crop]

    # Spalten mit sehr wenig Tinte gelten als Lücke (< 10 % des Maximums)
    gap_mask = profile < (0.10 * profile.max())
    min_gap = max(6, int(0.02 * w))  # Mindestbreite einer Lücke (2 % der Bildbreite)
    split_points: List[int] = []
    start = None
    for i, is_gap in enumerate(gap_mask):
        if is_gap and start is None:
            start = i  # Beginn einer Lücke
        if not is_gap and start is not None:
            if i - start >= min_gap:
                split_points.append((start + i) // 2)  # Mitte der Lücke als Trennpunkt
            start = None
    # Lücke am rechten Bildrand abschließen
    if start is not None and (len(gap_mask) - start) >= min_gap:
        split_points.append((start + len(gap_mask) - 1) // 2)

    if not split_points:
        return [crop]

    # Crop an den Trennpunkten aufteilen (Chunks < 12 px Breite ignorieren)
    chunks, x_prev = [], 0
    for x_cut in split_points + [w]:
        if (x_cut - x_prev) >= 12:
            part = crop[:, x_prev:x_cut]
            if part.size > 0:
                chunks.append(part)
        x_prev = x_cut

    # Zu viele Chunks deuten auf Oversegmentierung hin → Original zurückgeben
    return [crop] if len(chunks) > 6 or len(chunks) == 0 else chunks


def recognize_handwriting(image_crop: np.ndarray, processor, model, device,
                           dictionary_score_fn, max_new_tokens: int = 24) -> str:
    """Erkennt Handschrift in einem Bild-Crop mit TrOCR.

    Erzeugt mehrere Bildvarianten (via preprocess_variants), führt TrOCR-Inferenz
    für jede durch und wählt den Kandidaten mit dem besten deutschen Wörterbuch-Score
    aus. Bei Gleichstand gewinnt der längere Text.

    TrOCR-Parameter:
    - num_beams=8: Beam-Search mit 8 Pfaden
    - length_penalty=1.2: bevorzugt längere Ausgaben leicht
    - no_repeat_ngram_size=3: verhindert Wiederholungen von 3-Grammen
    - repetition_penalty=1.3: bestraft wiederholte Token

    Args:
        image_crop:          Ausgeschnittenes Bild (BGR, NumPy-Array).
        processor:           TrOCRProcessor (Tokenizer + Feature-Extraktor).
        model:               VisionEncoderDecoderModel (TrOCR).
        device:              "cuda" oder "cpu".
        dictionary_score_fn: Funktion (str) -> float zur Kandidatenbewertung.
        max_new_tokens:      Maximale Ausgabelänge in Tokens.

    Returns:
        Erkannter Text als String, oder "" bei Fehler/leerem Crop.
    """
    if image_crop is None or image_crop.size == 0:
        return ""
    h, w = image_crop.shape[:2]
    if h < 5 or w < 5:
        return ""

    candidates: List[str] = []
    for variant in preprocess_variants(image_crop):
        # BGR -> RGB, da TrOCR auf RGB-Bilder trainiert ist
        image = Image.fromarray(cv2.cvtColor(variant, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            length_penalty=1.2,
            num_beams=8,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            early_stopping=True,
        )
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if text:
            candidates.append(text)

    if not candidates:
        return ""
    # Besten Kandidaten nach DE-Wörterbuch-Score wählen; Länge als Tiebreaker
    return sorted(candidates, key=lambda t: (dictionary_score_fn(t), len(t)), reverse=True)[0]


def detect_text_regions(img_path: str, reader) -> Tuple[List[tuple], str]:
    """Erkennt Textregionen im Bild mit EasyOCR in Normal- und Strikt-Modus.

    Beide Modi werden ausgeführt und der Modus mit mehr erkannten Boxen gewinnt
    (Strikt muss mindestens 1 Box mehr liefern). Das Bild wird vor der Erkennung
    gitter-normalisiert, damit Karopapier-Hintergrund nicht stört.

    Modi:
    - normal:       text_threshold=0.25, low_text=0.25 – robuste Erkennung
    - strict-word:  text_threshold=0.20, low_text=0.20, engere Zeilengrenzen –
                    trennt dicht stehende Wörter besser auf

    Args:
        img_path: Pfad zur Bilddatei.
        reader:   Initialisierter EasyOCR-Reader.

    Returns:
        Tuple aus (Liste von (bbox, text, konfidenz), Modus-String).
    """
    img = cv2.imread(img_path)
    clean = remove_grid_background(img) if img is not None else None
    # Gitter-normalisiertes Bild bevorzugen; Fallback auf Originalpfad
    src = clean if clean is not None else img_path

    normal = reader.readtext(src, detail=1, paragraph=False,
                             text_threshold=0.25, low_text=0.25,
                             contrast_ths=0.05, adjust_contrast=0.7)
    strict = reader.readtext(src, detail=1, paragraph=False,
                             text_threshold=0.2, low_text=0.2,
                             contrast_ths=0.05, adjust_contrast=0.7,
                             width_ths=0.0, ycenter_ths=0.3, height_ths=0.3)
    # Strikt-Modus nur verwenden, wenn er deutlich mehr Regionen liefert
    if len(strict) >= len(normal) + 1:
        return strict, "strict-word-split"
    return normal, "normal"


def sort_regions_reading_order(results: List[tuple]) -> List[tuple]:
    """Sortiert Textregionen nach westlicher Lesereihenfolge (oben→unten, links→rechts).

    Ablauf:
    1. Mittelpunkt jeder Bounding-Box berechnen.
    2. Regionen mit ähnlichem y-Mittelpunkt (innerhalb line_thresh) werden zu einer
       Textzeile zusammengefasst. line_thresh = 60 % des Median der Boxhöhen.
    3. Innerhalb jeder Zeile wird nach x-Koordinate (links→rechts) sortiert.
    4. Zeilen werden von oben nach unten ausgegeben.

    Der laufende Zeilenmittelpunkt wird exponentiell geglättet (α=0.3), damit
    Schrägen im Text nicht zu falschen Zeilenwechseln führen.

    Args:
        results: Liste von EasyOCR-Ergebnissen (bbox, text, konfidenz).

    Returns:
        Sortierte Liste in Lesereihenfolge.
    """
    if not results:
        return results

    enriched, heights = [], []
    for r in results:
        xs = [pt[0] for pt in r[0]]
        ys = [pt[1] for pt in r[0]]
        h = max(1.0, max(ys) - min(ys))
        heights.append(h)
        # (original, x_min, y_min, y_center, height) für spätere Sortierung
        enriched.append((r, min(xs), min(ys), (max(ys) + min(ys)) / 2.0, h))

    # Schwellwert für Zeilenzugehörigkeit: 60 % der mittleren Boxhöhe
    line_thresh = max(10.0, 0.6 * float(np.median(heights)))
    # Vorläufig nach y-Mittelpunkt sortieren, um Zeilen von oben nach unten zu verarbeiten
    enriched.sort(key=lambda t: t[3])

    lines, current_line, current_y = [], [], None
    for item in enriched:
        y_center = item[3]
        if current_y is None or abs(y_center - current_y) <= line_thresh:
            # Region gehört zur aktuellen Zeile
            current_line.append(item)
            # Exponentiell geglätteter Zeilenmittelpunkt (α=0.3)
            current_y = y_center if current_y is None else (0.7 * current_y + 0.3 * y_center)
        else:
            # Neue Zeile beginnen
            lines.append(current_line)
            current_line, current_y = [item], y_center
    if current_line:
        lines.append(current_line)

    sorted_results: List[tuple] = []
    for line in lines:
        # Innerhalb jeder Zeile: links→rechts nach x_min sortieren
        line.sort(key=lambda t: t[1])
        sorted_results.extend(t[0] for t in line)
    return sorted_results
