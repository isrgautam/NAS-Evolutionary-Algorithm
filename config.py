
import torch

DATASET_PATH = './PlantVillage'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
BATCH_SIZE = 32

STAGES_CONFIG = [
    (16, [1, 2], 1),
    (24, [2, 3], 2),
    (32, [2, 3, 4], 2),
    (64, [2, 3, 4], 2),
    (96, [2, 3], 1),
]

NUM_OP_CHOICES = 4

# --- Evolutionary Search ---
POPULATION_SIZE = 10
NUM_GENERATIONS = 5
PROXY_EPOCHS = 3
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.5

# --- Fitness Function Weights ---
# The core of being "hardware-aware". How much to penalize large models.
# fitness = accuracy - (lambda_macs * (macs_proxy))
LAMBDA_MACS = 0.01

# --- Final Model Training ---
FINAL_MODEL_EPOCHS = 20
LEARNING_RATE = 0.001

