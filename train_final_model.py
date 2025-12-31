"""
Main script to train the best-found architecture from scratch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from config import *
from data_loader import get_dataloaders
from search_space import SuperNet
from utils import load_architecture

def main():
    print(f"--- Training Final Discovered Model on {DEVICE} ---")
    
    if not os.path.exists('best_architecture.json'):
        print("Error: 'best_architecture.json' not found. Please run 'run_search.py' first.")
        return
        
    best_arch = load_architecture('best_architecture.json')
    print(f"Loaded best architecture: {best_arch}")
    
    train_loader, val_loader, num_classes = get_dataloaders(DATASET_PATH)
    final_model = SuperNet(num_classes=num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    print("\nStarting final training...")
    for epoch in range(FINAL_MODEL_EPOCHS):
        final_model.train()
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{FINAL_MODEL_EPOCHS} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = final_model(inputs, best_arch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        final_model.eval()
        corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = final_model(inputs, best_arch)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
        
        val_acc = (corrects.double() / len(val_loader.dataset)).item()
        print(f"Epoch {epoch+1}/{FINAL_MODEL_EPOCHS} -> Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(final_model.state_dict(), 'final_model_best_weights.pth')
            print(f"  -> New best model saved with accuracy: {val_acc:.4f}")

    print("\n--- Final Training Finished ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("Model weights saved to 'final_model_best_weights.pth'")

if __name__ == '__main__':
    main()

