"""
Contains the logic for the Evolutionary Algorithm that will
search for the optimal network architecture.
"""
import random
from config import STAGES_CONFIG, NUM_OP_CHOICES

class EvolutionarySearcher:
    def generate_random_architecture(self):
        """Generates a single random architecture encoding."""
        architecture = []
        for _, depth_choices, _ in STAGES_CONFIG:
            depth = random.choice(depth_choices)
            stage_ops = [random.randint(0, NUM_OP_CHOICES - 1) for _ in range(depth)]
            architecture.append(stage_ops)
        return architecture

    def mutate_architecture(self, architecture, mutation_prob):
        """Applies mutation to an architecture encoding."""
        mutated_arch = []
        for stage_idx, stage_ops in enumerate(architecture):
            # Mutate operators
            mutated_stage_ops = [
                random.randint(0, NUM_OP_CHOICES - 1) if random.random() < mutation_prob else op
                for op in stage_ops
            ]
            
            # Mutate depth
            if random.random() < mutation_prob:
                depth_choices = STAGES_CONFIG[stage_idx][1]
                new_depth = random.choice(depth_choices)
                if new_depth > len(mutated_stage_ops):
                    mutated_stage_ops.extend([random.randint(0, NUM_OP_CHOICES - 1) for _ in range(new_depth - len(mutated_stage_ops))])
                else:
                    mutated_stage_ops = mutated_stage_ops[:new_depth]
            
            mutated_arch.append(mutated_stage_ops)
        return mutated_arch

    def crossover_architectures(self, arch1, arch2):
        """Performs crossover between two parent architectures."""
        child_arch = []
        for stage_idx in range(len(STAGES_CONFIG)):
            child_arch.append(random.choice([arch1[stage_idx], arch2[stage_idx]]))
        return child_arch

