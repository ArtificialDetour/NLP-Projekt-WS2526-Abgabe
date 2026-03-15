import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import TRAIN_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE

def get_dataloaders():
    """Erstellt Trainings- und Test-Dataloaders für die UML-Komponenten.

    Nutzt torchvision ImageFolder: Klassennamen werden automatisch aus den
    Unterordnernamen abgeleitet.

    Returns:
        Tuple aus (Dataloaders-Dict, Klassennamen-Liste, Dataset-Größen-Dict).
    """
    # ViT-Standardnormalisierung: Pixelwerte auf [-1, 1] skalieren
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),  # Einfache Augmentierung zur Regularisierung
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize
        ]),
    }

    image_datasets = {
        'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
        'test': datasets.ImageFolder(TEST_DIR, data_transforms['test'])
    }

    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        ),
        'test': DataLoader(
            image_datasets['test'],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )
    }

    class_names = image_datasets['train'].classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    return dataloaders, class_names, dataset_sizes

if __name__ == '__main__':
    # Schnelltest: Dataloader und Batch-Dimensionen prüfen
    dataloaders, class_names, sizes = get_dataloaders()
    print(f"Classes: {class_names}")
    print(f"Dataset sizes: {sizes}")
    inputs, classes = next(iter(dataloaders['train']))
    print(f"Batch inputs shape: {inputs.shape}")
    print(f"Batch classes shape: {classes.shape}")
# done