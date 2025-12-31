"""
Main script to execute the Neural Architecture Search.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from config import *
from data_loader import get_dataloaders
from search_space import SuperNet
from search_strategy import EvolutionarySearcher
from utils import train_one_epoch, validate_one_epoch, count_macs_proxy, get_hardware_aware_fitness, save_architecture

def evaluate_architecture(arch, supernet, train_loader, val_loader):
    """Evaluates a single architecture's fitness."""
    print(f"\nEvaluating Architecture: {arch}")
    
    criterion = nn.CrossEntropyLoss()
    # Use a fresh optimizer for each evaluation to be fair
    optimizer = optim.Adam(supernet.parameters(), lr=0.001)

    for epoch in range(PROXY_EPOCHS):
        print(f"  Proxy Epoch {epoch+1}/{PROXY_EPOCHS}")
        train_one_epoch(supernet, train_loader, criterion, optimizer, arch)
    
    accuracy = validate_one_epoch(supernet, val_loader, criterion, arch)
    macs = count_macs_proxy(arch)
    fitness = get_hardware_aware_fitness(accuracy, macs)

    print(f"  Result -> Accuracy: {accuracy:.4f}, MACs-proxy: {macs}, Fitness: {fitness:.4f}")
    return fitness

def main():
    print(f"--- Starting Neural Architecture Search on {DEVICE} ---")
    train_loader, val_loader, num_classes = get_dataloaders(DATASET_PATH)
    supernet = SuperNet(num_classes=num_classes).to(DEVICE)
    searcher = EvolutionarySearcher()

    population = [searcher.generate_random_architecture() for _ in range(POPULATION_SIZE)]
    fitness_scores = np.zeros(POPULATION_SIZE)

    for generation in range(NUM_GENERATIONS):
        print(f"\n{'='*20} Generation {generation+1}/{NUM_GENERATIONS} {'='*20}")
        
        for i, arch in enumerate(population):
            fitness_scores[i] = evaluate_architecture(arch, supernet, train_loader, val_loader)

        # Select the top 50% of the population as parents for the next generation
        parent_indices = np.argsort(fitness_scores)[- (POPULATION_SIZE // 2):]
        parents = [population[i] for i in parent_indices]
        
        # Create the next generation
        next_generation = parents[:]
        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)
            
            if random.random() < CROSSOVER_PROBABILITY:
                child = searcher.crossover_architectures(parent1, parent2)
            else:
                child = random.choice(parents) # Asexual reproduction
            
            child = searcher.mutate_architecture(child, MUTATION_PROBABILITY)
            next_generation.append(child)
            
        population = next_generation

    print("\n--- Search Finished. Evaluating final population... ---")
    for i, arch in enumerate(population):
        fitness_scores[i] = evaluate_architecture(arch, supernet, train_loader, val_loader)
        
    best_arch_index = np.argmax(fitness_scores)
    best_architecture = population[best_arch_index]
    
    print(f"\nBest Architecture Found: {best_architecture}")
    print(f"With Final Fitness Score: {fitness_scores[best_arch_index]:.4f}")
    
    save_architecture('best_architecture.json', best_architecture)
    print("Best architecture saved to 'best_architecture.json'")

if __name__ == '__main__':
    main()

