import os
import torch
import torch.nn as nn
from torch.optim import AdamW

from config import NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, WEIGHTS_DIR
from dataset import get_dataloaders
from model import UMLComponentClassifier

def train_model():
    print("Loading data...")
    dataloaders, class_names, dataset_sizes = get_dataloaders()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing model...")
    model = UMLComponentClassifier()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_acc = 0.0

    print("Beginning training...")
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        # Jede Epoche: Training + Validierung auf dem Test-Set
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Gradienten zurücksetzen (akkumulieren sich sonst über Batches)
                optimizer.zero_grad()

                # Vorwärtsdurchlauf; Gradient nur im Training berechnen
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Verlust und Korrektklassifikationen akkumulieren
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Bestes Modell nach Validierungsgenauigkeit tracken (für das finale Log)
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc

            # Das letzte Modell am Ende der Test-Phase speichern
            if phase == 'test':
                last_model_weights_path = os.path.join(WEIGHTS_DIR, "best_vit_model.pth")
                torch.save(model.state_dict(), last_model_weights_path)
                print(f"Saved last model to {last_model_weights_path} (Acc: {epoch_acc:.4f})")

        print()

    print(f'Best test Acc: {best_acc:4f}')
    print('Training complete.')

if __name__ == '__main__':
    train_model()
