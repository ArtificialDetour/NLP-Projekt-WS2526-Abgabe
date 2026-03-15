import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from config import MODEL_NAME, NUM_CLASSES

class UMLComponentClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES):
        super(UMLComponentClassifier, self).__init__()

        # Vortrainierter ViT: Klassifikationskopf auf NUM_CLASSES angepasst
        # (ignore_mismatched_sizes nötig, da Vortraining auf 1000 ImageNet-Klassen)
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

if __name__ == "__main__":
    # Schnelltest: Modellinstanziierung und Forward-Pass prüfen
    model = UMLComponentClassifier()
    print("Model instantiated successfully.")

    # Dummy-Eingabe: Batch mit 2 Bildern (3×224×224)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    print(f"Output shape: {logits.shape}")  # Erwartet: [2, 5]
