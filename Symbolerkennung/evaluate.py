import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import WEIGHTS_DIR, OUTPUT_DIR, CLASSES
from dataset import get_dataloaders
from model import UMLComponentClassifier

def evaluate_model():
    print("Loading test data...")
    dataloaders, class_names, _ = get_dataloaders()
    test_loader = dataloaders['test']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UMLComponentClassifier()
    model_path = os.path.join(WEIGHTS_DIR, "best_vit_model.pth")
    
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        try:
           model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except Exception as e:
            print(f"Failed to load weights: {e}")
    else:
        print("No trained weights found. Evaluating with random initialization.")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Klassifikationsbericht berechnen
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)

    print("\nClassification Report:")
    print(report_text)

    # Ergebnisse als Markdown-Tabelle speichern
    results_path = os.path.join(OUTPUT_DIR, "results.md")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("# UML Component Recognition Evaluation Results\n\n")
        f.write("## Classification Metrics per Component\n\n")

        f.write("| Component | Precision | Recall | F1-Score | Support |\n")
        f.write("|-----------|-----------|--------|----------|---------|\n")

        for cls in class_names:
            metrics = report[cls]
            f.write(f"| {cls} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {metrics['support']} |\n")

        # Makro-Durchschnitt über alle Klassen
        macro = report['macro avg']
        f.write(f"| **Macro Avg** | **{macro['precision']:.4f}** | **{macro['recall']:.4f}** | **{macro['f1-score']:.4f}** | **{macro['support']}** |\n")

    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    evaluate_model()
