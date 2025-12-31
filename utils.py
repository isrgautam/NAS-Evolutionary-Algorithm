"""
Contains reusable helper functions for training, validation,
hardware-aware metric calculation, and saving results.
"""
import torch
import json
from tqdm import tqdm
from config import DEVICE, LAMBDA_MACS

def train_one_epoch(model, dataloader, criterion, optimizer, arch):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs, arch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def validate_one_epoch(model, dataloader, criterion, arch):
    model.eval()
    corrects = 0
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs, arch)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
    acc = corrects.double() / len(dataloader.dataset)
    return acc.item()

def count_macs_proxy(architecture):
    """A simple proxy for MACs, counting the number of blocks."""
    macs = 0
    for stage_arch in architecture:
        macs += len(stage_arch)
    return macs

def get_hardware_aware_fitness(accuracy, macs):
    """Calculates fitness score, penalizing for high MACs."""
    mac_penalty = macs / 20.0 # Normalize proxy to a reasonable scale
    return accuracy - (LAMBDA_MACS * mac_penalty)

def save_architecture(filepath, architecture):
    """Saves a single architecture to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(architecture, f, indent=4)

def load_architecture(filepath):
    """Loads a single architecture from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

