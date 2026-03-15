import math
from spellchecker import SpellChecker
from typing import Tuple

# Globale Rechtschreibprüfer für Deutsch und Englisch
spell_de = SpellChecker(language='de')
spell_en = SpellChecker(language='en')

# Satzzeichen, die beim Wortvergleich abgestreift werden
PUNCTUATION = ".,!?()[]:{}\"'"
# Vokalmengen für die Namenerkennung (Heuristik)
_VOWELS = set('aeiouäöüAEIOUÄÖÜ')

# --- German BERT (lazy geladen) ---
_bert_lm_pipe = None

def get_bert_lm():
    """Laedt dbmdz/bert-base-german-cased lazy (~440 MB, einmalig)."""
    global _bert_lm_pipe
    if _bert_lm_pipe is None:
        try:
            from transformers import pipeline as hf_pipeline
            print("Lade German-BERT fuer LM-Scoring (dbmdz/bert-base-german-cased)...")
            _bert_lm_pipe = hf_pipeline("fill-mask", model="dbmdz/bert-base-german-cased",
                                         top_k=50, device=-1)
            print("BERT bereit.")
        except Exception as e:
            print(f"[BERT-LM] Nicht verfuegbar: {e}")
            _bert_lm_pipe = False
    return _bert_lm_pipe if _bert_lm_pipe else None


def lm_sentence_score(sentence: str) -> float:
    """Berechnet einen Pseudo-Log-Likelihood-Score eines Satzes mit German BERT.

    Jedes Wort wird einzeln maskiert und die BERT-Vorhersage-Wahrscheinlichkeit
    für das Original-Wort als log(p) addiert. Der Score wird auf die Wortanzahl
    normiert, damit unterschiedlich lange Sätze vergleichbar bleiben.

    Args:
        sentence: Eingabesatz als String.

    Returns:
        Durchschnittlicher log-Wahrscheinlichkeitsscore (<= 0.0).
        0.0 bedeutet, dass BERT nicht verfügbar oder der Satz zu kurz ist.
    """
    words = sentence.split()
    if len(words) < 2:
        return 0.0
    pipe = get_bert_lm()
    if pipe is None:
        return 0.0
    # Sentence-Case: Grossschreibung mitten im Satz normalisieren (OCR-Artefakt).
    normalized = [words[0]] + [
        w.lower() if w[0].isupper() and w[1:].islower() else w for w in words[1:]
    ]
    total, count = 0.0, 0
    for i, word in enumerate(normalized):
        # Jedes Wort wird einmal maskiert, um seine Kontextwahrscheinlichkeit zu messen.
        masked_text = " ".join(normalized[:i] + ["[MASK]"] + normalized[i + 1:])
        try:
            preds = pipe(masked_text)
            word_lower = word.lower().strip(PUNCTUATION)
            # Walrus-Operator: Falls das Original-Wort unter den Top-k-Vorhersagen ist,
            # wird sein Score direkt in total akkumuliert.
            found = any(p["token_str"].strip().lower() == word_lower and
                        (total := total + math.log(max(p["score"], 1e-10))) is not None
                        for p in preds)
            if not found:
                # Wort nicht in Top-k: sehr kleiner Fallback-Score (log 1e-6 ≈ -13.8)
                total += math.log(1e-6)
            count += 1
        except Exception:
            pass
    return total / max(count, 1)


def levenshtein_distance(a: str, b: str) -> int:
    """Berechnet die Levenshtein-Editierdistanz zwischen zwei Zeichenketten.

    Verwendet die iterative Wagner-Fischer-Methode mit zwei Zeilen (O(n) Speicher).
    Erlaubte Operationen: Einfügen, Löschen, Ersetzen (je Kosten 1).

    Args:
        a: Erste Zeichenkette.
        b: Zweite Zeichenkette.

    Returns:
        Minimale Anzahl an Editieroperationen, um a in b umzuwandeln.
    """
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    # prev enthält die Kosten der vorherigen Zeile (Präfixe von b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]  # Kosten für leeres b-Präfix = i Löschungen
        for j, cb in enumerate(b, 1):
            # min aus: Löschen (curr[j-1]+1), Einfügen (prev[j]+1), Ersetzen/Übereinstimmung
            curr.append(min(curr[j-1]+1, prev[j]+1, prev[j-1]+(0 if ca==cb else 1)))
        prev = curr
    return prev[-1]


def correct_text(text: str) -> str:
    """Korrigiert OCR-Fehler wortweise mit deutschem und englischem Wörterbuch.

    Für jedes Wort wird zunächst das Umfeld (Satzzeichen) abgetrennt. Wörter,
    die bereits im deutschen Wörterbuch stehen, werden unverändert gelassen.
    Kurze Namen (TitleCase, 3–4 Zeichen) werden ebenfalls nicht angepasst.
    Korrekturen, die die Wortlänge um mehr als 2 Zeichen verändern, werden
    verworfen, um zu aggressive Korrekturen zu vermeiden.

    Args:
        text: Rohtext aus der OCR-Erkennung.

    Returns:
        Korrigierter Text als String.
    """
    if not text:
        return text
    corrected = []
    for word in text.split():
        # Führende/abschließende Satzzeichen separat speichern
        prefix, suffix, stripped = "", "", word
        while stripped and not stripped[0].isalpha():
            prefix += stripped[0]; stripped = stripped[1:]
        while stripped and not stripped[-1].isalpha():
            suffix = stripped[-1] + suffix; stripped = stripped[:-1]

        # Zu kurze Wörter, Zahlen oder Unterstriche unverändert lassen
        if len(stripped) < 2 or any(c.isdigit() for c in stripped) or "_" in stripped:
            corrected.append(word); continue

        token = stripped.lower()
        in_de = token not in spell_de.unknown([token])
        in_en = token not in spell_en.unknown([token])

        if in_de:
            corrected.append(word); continue

        # Echte Namen (3-4 Zeichen, Vokal, TitleCase) nicht anfassen.
        has_vowel = any(c in _VOWELS for c in stripped)
        if stripped[0].isupper() and stripped[1:].islower() and 3 <= len(stripped) <= 4 and has_vowel:
            corrected.append(word); continue

        correction = None
        if in_en and not in_de:
            # Wort ist englisch, aber nicht deutsch: nur akzeptieren, wenn DE-Korrektur sehr ähnlich ist
            de_c = spell_de.correction(token)
            correction = de_c if de_c and levenshtein_distance(token, de_c) <= 1 else token
        else:
            # Weder DE noch EN: beste verfügbare Korrektur verwenden
            correction = spell_de.correction(token) or spell_en.correction(token)

        # Korrekturen, die das Wort stark verlängern/verkürzen, ablehnen
        if correction and abs(len(correction) - len(token)) > 2:
            correction = None

        if correction:
            if stripped[0].isupper():
                correction = correction.capitalize()
            corrected.append(prefix + correction + suffix)
        else:
            corrected.append(word)
    return " ".join(corrected)


def dictionary_score(text: str) -> float:
    """Berechnet den Anteil bekannter deutschen Wörter im Text.

    Zählt nur Wörter mit mindestens 2 Zeichen (nach Satzzeichenbereinigung).
    Gibt einen Wert zwischen 0.0 (kein Wort bekannt) und 1.0 (alle bekannt) zurück.

    Args:
        text: Zu bewertender Text.

    Returns:
        Anteil der im deutschen Wörterbuch gefundenen Wörter als float.
    """
    words = [w.strip(PUNCTUATION) for w in text.split() if len(w.strip(PUNCTUATION)) >= 2]
    if not words:
        return 0.0
    known = sum(1 for w in words if w.lower() not in spell_de.unknown([w.lower()]))
    return known / len(words)


def best_text(easyocr_text: str, easyocr_conf: float, trocr_corrected: str) -> Tuple[str, str]:
    """Wählt das bessere OCR-Ergebnis zwischen EasyOCR und TrOCR aus.

    Entscheidungskriterien (in Reihenfolge):
    1. Fallback auf das einzige nicht-leere Ergebnis.
    2. TrOCR gewinnt, wenn sein Wörterbuch-Score deutlich besser ist (>+0.20).
    3. EasyOCR gewinnt bei hoher Konfidenz (>=0.75) oder deutlich besserem Score.
    4. Kombinierte Gewichtung aus Konfidenz und TrOCR-Score.
    5. EasyOCR als letzter Fallback.

    Args:
        easyocr_text:    Rohtext aus EasyOCR.
        easyocr_conf:    Konfidenzwert von EasyOCR (0.0–1.0).
        trocr_corrected: Rechtschreibkorrigierter TrOCR-Text.

    Returns:
        Tuple aus (gewählter Text, Begründungs-String).
    """
    easy_score = dictionary_score(easyocr_text)
    trocr_score = dictionary_score(trocr_corrected)

    easy_clean = easyocr_text.strip()
    trocr_clean = trocr_corrected.strip()

    # Nur eines der Ergebnisse ist nicht leer
    if not easy_clean and trocr_clean:
        return trocr_corrected, "TrOCR (EasyOCR leer)"
    if not trocr_clean and easy_clean:
        return easyocr_text, "EasyOCR (TrOCR leer)"

    # TrOCR hat deutlich mehr deutsche Wörter erkannt
    if trocr_clean and trocr_score > easy_score + 0.20:
        return trocr_corrected, "TrOCR (dict besser)"

    # EasyOCR ist sehr konfident oder hat deutlich besseren Wörterbuch-Score
    if easy_clean and (easyocr_conf >= 0.75 or easy_score > trocr_score + 0.20):
        return easyocr_text, f"EasyOCR (conf={easyocr_conf:.2f})"

    # Schwellwert-basierte Kombination: Konfidenz + TrOCR-Score > 1.0
    if easyocr_conf + trocr_score > 1.0 and trocr_clean:
        return trocr_corrected, "TrOCR (Gewichtung)"

    return easyocr_text, "EasyOCR (Fallback)"


def bert_correct_sentence(sentence: str) -> str:
    """Ersetzt nicht-deutsche Wörter durch German-BERT-Vorhersagen im Kontext.

    Läuft max. 2 Durchläufe: Korrekturen verbessern den Kontext für nachfolgende
    Masken (iteratives Fill-Mask). Wörter, die bereits im deutschen Wörterbuch
    stehen oder Ziffern/Sonderzeichen enthalten, werden nicht angefasst.

    Die innere Funktion `is_plausible_replacement` filtert unplausible
    BERT-Kandidaten anhand von Levenshtein-Ähnlichkeit und Mindest-Score.

    Args:
        sentence: Eingabesatz (ggf. mit OCR-Fehlern).

    Returns:
        Korrigierter Satz als String.
    """
    words = sentence.split()
    if len(words) < 2:
        return sentence
    pipe = get_bert_lm()
    if pipe is None:
        return sentence

    # Häufige deutsche Funktionswörter, die als Ersatz für sehr kurze OCR-Fragmente erlaubt sind
    common_short_words = {
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "einen",
        "ich", "du", "er", "sie", "es", "wir", "ihr", "bin", "bist", "ist", "sind",
        "und", "im", "in", "zu", "von", "mit", "auf", "an"
    }

    def is_plausible_replacement(src: str, cand: str, score: float) -> bool:
        """Prüft, ob ein BERT-Kandidat als Ersatz für ein OCR-Wort akzeptabel ist.

        Args:
            src:   Original-OCR-Token (ohne Satzzeichen, Kleinbuchstaben).
            cand:  BERT-Kandidat (Kleinbuchstaben).
            score: BERT-Wahrscheinlichkeit des Kandidaten.

        Returns:
            True, wenn der Kandidat als Ersatz geeignet ist.
        """
        src_l = src.lower()
        cand_l = cand.lower()
        # Kandidat muss selbst ein gültiges deutsches Wort sein
        if cand_l in spell_de.unknown([cand_l]):
            return False

        dist = levenshtein_distance(src_l, cand_l)
        max_len = max(len(src_l), len(cand_l), 1)
        norm_dist = dist / max_len

        # Normalfall: nur aehnliche Ersetzungen akzeptieren.
        if norm_dist <= 0.45 and score >= 0.03:
            return True

        # Ausnahme fuer sehr kurze OCR-Fragmente: nur haeufige Funktionswoerter zulassen.
        if len(src_l) <= 2 and cand_l in common_short_words and score >= 0.15:
            return True

        return False

    corrected = list(words)
    for _pass in range(2):  # Maximal 2 Iterationen über den Satz
        any_change = False
        for i, word in enumerate(corrected):
            # Satzzeichen vom Wortkern trennen
            prefix, suffix, stripped = "", "", word
            while stripped and not stripped[0].isalpha():
                prefix += stripped[0]; stripped = stripped[1:]
            while stripped and not stripped[-1].isalpha():
                suffix = stripped[-1] + suffix; stripped = stripped[:-1]
            if len(stripped) < 2 or any(c.isdigit() for c in stripped) or "_" in stripped:
                continue
            token = stripped.lower()
            if token not in spell_de.unknown([token]):
                continue  # bereits gueltiges Deutsch
            # Wort maskieren und BERT nach dem wahrscheinlichsten Ersatz fragen
            masked = " ".join(corrected[:i] + ["[MASK]"] + corrected[i + 1:])
            try:
                preds = pipe(masked)
                for p in preds:
                    cand = p["token_str"].strip()
                    score = float(p.get("score", 0.0))
                    # Subword-Tokens (##...) und sehr kurze Kandidaten überspringen
                    if not cand or cand.startswith("##") or len(cand) < 2:
                        continue
                    if not is_plausible_replacement(stripped, cand, score):
                        continue
                    if stripped[0].isupper():
                        cand = cand.capitalize()
                    corrected[i] = prefix + cand + suffix
                    any_change = True
                    break  # Besten Kandidaten nehmen, dann weiter zum nächsten Wort
            except Exception:
                pass
        if not any_change:
            break  # Kein Wort mehr geändert: Konvergiert, Schleife abbrechen
    return " ".join(corrected)


def dictionary_score_de_only(text: str) -> float:
    """Alias für dictionary_score explizit für rein deutschen Kontext.

    Delegiert vollständig an dictionary_score(); existiert für semantische
    Klarheit an Aufrufstellen, die ausdrücklich nur deutsch bewerten wollen.

    Args:
        text: Zu bewertender Text.

    Returns:
        Anteil der im deutschen Wörterbuch gefundenen Wörter als float.
    """
    return dictionary_score(text)


def best_sentence_candidate(region_sentence: str, line_sentence: str) -> Tuple[str, str]:
    """Wählt den besseren Gesamtsatz aus Regions-Join und Zeilen-Crop (TrOCR).

    Entscheidungshierarchie:
    1. Wörterbuch-Score (DE): Klarer Sieger bei Differenz > 0.15.
       Vorher: ZeilenCrop-Score wird abgewertet, wenn er deutlich kürzer ist.
       Außerdem: ZeilenCrop wird übersprungen, wenn er kaum deutsch ist (<0.25).
    2. BERT-LM-Score: Tiebreaker bei ähnlichem Wörterbuch-Score (Differenz > 0.5).
    3. Fallback: Wer mehr deutsche Wörter hat, gewinnt.

    Args:
        region_sentence: Zusammengeführter Text aus einzelnen OCR-Regionen.
        line_sentence:   Text aus dem Zeilen-Crop (typisch TrOCR-Output).

    Returns:
        Tuple aus (gewählter Satz, Begründungs-String mit Scores).
    """
    r_score = dictionary_score(region_sentence)
    l_score = dictionary_score(line_sentence)

    # TrOCR-Zeilenoutput klar englisch? -> direkt Regionen-Ergebnis verwenden.
    if l_score < 0.25 and r_score >= l_score:
        return region_sentence, f"Regionen-Join (ZeilenCrop englisch, DE-dict={l_score:.2f})"

    # Laengere Ausgabe bevorzugen: kurze Crops werden abgewertet.
    r_words = len(region_sentence.split())
    l_words = len(line_sentence.split())
    if r_words > 0 and l_words < r_words * 0.6:
        # Proportionale Abwertung: weniger Wörter -> niedrigerer effektiver Score
        l_score *= l_words / r_words

    # Klarer Sieger nach Wörterbuch-Score
    if r_score > l_score + 0.15:
        return region_sentence, f"Regionen-Join (dict={r_score:.2f} > ZeilenCrop dict={l_score:.2f})"
    if l_score > r_score + 0.15:
        return line_sentence, f"ZeilenCrop (dict={l_score:.2f} > Regionen dict={r_score:.2f})"

    # Scores zu ähnlich: BERT-LM als Tiebreaker
    r_lm = lm_sentence_score(region_sentence)
    l_lm = lm_sentence_score(line_sentence)
    if r_lm != 0.0 or l_lm != 0.0:
        if l_lm > r_lm + 0.5:
            return line_sentence,   f"ZeilenCrop (LM={l_lm:.2f} > Regionen LM={r_lm:.2f}, dict gleich)"
        if r_lm > l_lm + 0.5:
            return region_sentence, f"Regionen-Join (LM={r_lm:.2f} > ZeilenCrop LM={l_lm:.2f}, dict gleich)"

    # Beide schwach: nimm was mehr deutsche Woerter hat.
    if r_score >= l_score:
        return region_sentence, f"Regionen-Join Fallback (dict={r_score:.2f})"
    return line_sentence, f"ZeilenCrop Fallback (dict={l_score:.2f})"
