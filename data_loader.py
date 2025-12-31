import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from config import BATCH_SIZE, NUM_WORKERS

def get_dataloaders(dataset_path):
    if not os.path.isdir(dataset_path) or not os.listdir(dataset_path):
        print(f"Error: Dataset not found at '{dataset_path}'.")
        print("Please download PlantVillage and update DATASET_PATH in config.py")
        exit()

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(dataset_path, transform=data_transform)
    num_classes = len(full_dataset.classes)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Dataset loaded with {num_classes} classes. Using {len(train_dataset)} for training, {len(val_dataset)} for validation.")
    return train_loader, val_loader, num_classes

