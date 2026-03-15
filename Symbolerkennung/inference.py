import os
import glob
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from config import INPUT_PARTS_DIR, INPUT_DIAGRAM_DIR, OUTPUT_DIR, WEIGHTS_DIR, CLASSES, IMAGE_SIZE
from model import UMLComponentClassifier
from graph_reconstruction import reconstruct_graph, export_to_mermaid

def load_model_and_transforms():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UMLComponentClassifier()
    model_path = os.path.join(WEIGHTS_DIR, "best_vit_model.pth")

    # Trainierte Gewichte laden, falls vorhanden
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        try:
           model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Failed to load weights properly: {e}\nUsing randomly initialized model for inference test run.")
    else:
        print("No trained weights found. Evaluating with random initialization.")

    model = model.to(device)
    model.eval()

    # Validierungs-Transforms identisch zu dataset.py (Test-Set)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    return model, transform, device

def process_parts(model, transform, device):
    print(f"\nScanning '{INPUT_PARTS_DIR}' for single component images...")
    predictions_log = []
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}

    for filename in os.listdir(INPUT_PARTS_DIR):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_extensions:
            continue

        file_path = os.path.join(INPUT_PARTS_DIR, filename)
        print(f"Processing parts image: {filename}")

        try:
            image = Image.open(file_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)

            predicted_class_name = CLASSES[preds.item()]
            print(f"  -> Predicted class: {predicted_class_name}")

            predictions_log.append(f"{filename} --> Prediction: {predicted_class_name}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Nummerierte Metriken-Datei für diesen Lauf speichern
    if predictions_log:
        run_files = glob.glob(os.path.join(OUTPUT_DIR, "metrics_run_*.md"))
        run_numbers = []
        for f in run_files:
            try:
                basename = os.path.basename(f)
                num_str = basename.replace("metrics_run_", "").replace(".md", "")
                run_numbers.append(int(num_str))
            except ValueError:
                continue
        
        next_run = max(run_numbers) + 1 if run_numbers else 1
        metrics_file_path = os.path.join(OUTPUT_DIR, f"metrics_run_{next_run}.md")
        
        with open(metrics_file_path, "w", encoding="utf-8") as f:
            f.write(f"# Inference Predictions (Parts) - Run {next_run}\n\n")
            for log_entry in predictions_log:
                f.write(f"{log_entry}\n")
                
        print(f"Predictions saved to {metrics_file_path}")

def extract_nodes_from_diagram(image_path):
    # OpenCV-Pipeline: Konturen der UML-Symbole als Bounding-Boxes extrahieren
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Schwellwert + Dilation: kleine Lücken in Konturen schließen
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    # Kleines Rauschen herausfiltern (Mindestfläche 400 px²)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area > 400:
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)
            bounding_boxes.append((x1, y1, x2, y2))

    # Bounding-Boxes nach Lesereihenfolge sortieren (oben→unten, links→rechts)
    bounding_boxes.sort(key=lambda b: (b[1], b[0]))
    return bounding_boxes, img

def process_diagrams(model, transform, device):
    print(f"\nScanning '{INPUT_DIAGRAM_DIR}' for full diagram images...")
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    
    for filename in os.listdir(INPUT_DIAGRAM_DIR):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in image_extensions:
            continue
            
        file_path = os.path.join(INPUT_DIAGRAM_DIR, filename)
        print(f"\nProcessing full diagram: {filename}")
        
        try:
            boxes, raw_cv2_image = extract_nodes_from_diagram(file_path)
            if not boxes:
                print(f"No components detected in {filename}.")
                continue
                
            print(f"  -> Extracted {len(boxes)} potential nodes.")
            
            components = []
            predictions_log = []
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                # Ausschnitt als PIL-Bild für die Transform-Pipeline
                crop = raw_cv2_image[y1:y2, x1:x2]
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                input_tensor = transform(crop_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, preds = torch.max(outputs, 1)

                pred_class = CLASSES[preds.item()]
                node_id = i + 1

                components.append({
                    'id': node_id,
                    'type': pred_class,
                    'label': f'Node {node_id}'
                })
                print(f"     Node {node_id} predicted as: {pred_class}")
                predictions_log.append(f"Node {node_id} (x:{x1},y:{y1}) --> Prediction: {pred_class}")

            # Start- und End-Knoten einfordern (UML-Konvention)
            has_start = any(c['type'] == 'start' for c in components)
            has_ending = any(c['type'] == 'ending' for c in components)
            
            if not has_start:
                components.insert(0, {'id': 0, 'type': 'start', 'label': 'Start'})
                print("  -> Injected missing 'start' node.")
                
            if not has_ending:
                next_id = max((c['id'] for c in components), default=0) + 1
                components.append({'id': next_id, 'type': 'ending', 'label': 'End'})
                print("  -> Injected missing 'ending' node.")

            # Heuristische Kantenextraktion nach UML-Eingangs-/Ausgangsgrad-Regeln
            relations = []
            open_sources = []
            final_structural_components = []
            pending_labels = []

            for comp in components:
                c_type = comp['type']
                c_id = comp['id']

                # 'action'-Knoten wird als Kantenbeschriftung behandelt, nicht als Strukturknoten
                if c_type == 'action':
                    label = comp.get('text_label') or comp.get('label') or f"Action {c_id}"
                    pending_labels.append(label)
                    continue

                # Strukturknoten: wird im Graphen als Knoten hinzugefügt
                final_structural_components.append(comp)

                # Eingehende Verbindung auflösen
                if c_type != 'start':
                    if open_sources:
                        src_id = open_sources.pop(0)

                        # Kantenbeschriftung bestimmen
                        src_comp = next((c for c in components if c['id'] == src_id), None)
                        rel_label_parts = []

                        # Ausstehende 'action'-Beschriftungen übernehmen
                        if pending_labels:
                            rel_label_parts.append(", ".join(pending_labels))
                            pending_labels = []

                        # Verzweigungsknoten: erste Kante 'yes', zweite 'no'
                        if src_comp and src_comp['type'] == 'choice':
                            existing_out = sum(1 for (u, v, lbl) in relations if u == src_id)
                            branch = 'yes' if existing_out == 0 else 'no'
                            if rel_label_parts:
                                rel_label = f"{branch} ({rel_label_parts[0]})"
                            else:
                                rel_label = branch
                        else:
                            rel_label = rel_label_parts[0] if rel_label_parts else ''

                        relations.append((src_id, c_id, rel_label))

                # Ausgehende Verbindungsslots bereitstellen
                if c_type == 'start':
                    open_sources.append(c_id)  # 1 ausgehend
                elif c_type == 'state':
                    open_sources.append(c_id)  # 1 ausgehend
                elif c_type == 'choice':
                    open_sources.append(c_id)  # 1. Ausgang
                    open_sources.append(c_id)  # 2. Ausgang
                elif c_type == 'ending':
                    pass  # kein Ausgang
                
            graph = reconstruct_graph(final_structural_components, relations)
            output_md_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + "_diagram.md")
            export_to_mermaid(graph, output_md_path)
            
            # Vorhersageprotokoll für dieses Diagramm speichern
            with open(os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + "_details.md"), "w") as detail_f:
                detail_f.write(f"# Diagram Components: {filename}\n\n")
                for log in predictions_log:
                    detail_f.write(f"- {log}\n")
                
                detail_f.write("\n## Extracted Relations (Heuristic Sequence)\n")
                for u, v, _ in relations:
                    detail_f.write(f"- Node {u} is sequentially connected to Node {v}\n")
                
                detail_f.write(f"\nGraph successfully parsed to [{os.path.basename(output_md_path)}]({os.path.basename(output_md_path)})\n")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model, transform, device = load_model_and_transforms()
    
    if os.path.exists(INPUT_PARTS_DIR):
        process_parts(model, transform, device)
    
    if os.path.exists(INPUT_DIAGRAM_DIR):
        process_diagrams(model, transform, device)
        
    print("\nInference pipeline completed.")
