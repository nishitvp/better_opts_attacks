import torch
import transformers
import gc
import typing
import peft
import datasets
import random
import copy
import sys
import time
import pickle
import threading
import queue

import utils.attack_utility as attack_utility
import utils.experiment_logger as experiment_logger
from secalign_refactored import secalign

import torch
import math

class AggressiveRandomStrategy:
    def __init__(self, total_steps, max_mutations, decay_rate):
        """
        Args:
            total_steps: The expected total number of optimization iterations.
            max_mutations: Maximum number of tokens to flip simultaneously at step 0.
            decay_rate: How quickly to transition to single-token flips. 
                        Higher = faster transition to stable refinement.
        """
        self.total_steps = total_steps
        self.max_mutations = max_mutations
        self.decay_rate = decay_rate
        self.current_step = 0

    def __call__(self, tokenizer, best_tokens_indices, input_tokenized_data_list, substitution_validity_function, max_candidate_size):
        # 1. Determine the "Aggression Level" (Mutation Count)
        # Uses exponential decay: starts at max_mutations, tapers to 1.
        progress = self.current_step / self.total_steps
        mutation_count = max(1, round(self.max_mutations * math.exp(-self.decay_rate * progress)))
        
        indices_to_sample_batches = [] # List of sets containing (first_coord, second_coord)

        # 2. Candidate Generation Loop
        while len(indices_to_sample_batches) < max_candidate_size:
            current_candidate_mutations = set()
            
            # Pick 'mutation_count' unique coordinates to flip for this candidate
            # best_tokens_indices.shape[0] is the length of the optimized prompt segment
            attempts = 0
            while len(current_candidate_mutations) < mutation_count and attempts < 100:
                first_coord = torch.randint(0, best_tokens_indices.shape[0], (1,)).item()
                second_coord = torch.randint(0, best_tokens_indices.shape[1], (1,)).item()
                current_candidate_mutations.add((first_coord, second_coord))
                attempts += 1

            # 3. Validate the entire multi-token perturbation
            all_substitutions_valid = True
            for input_tokenized_data in input_tokenized_data_list:
                masks_data = input_tokenized_data["masks"]
                optim_mask = masks_data["optim_mask"]
                
                # Create a trial version of the tokens with ALL proposed swaps
                trial_tokens = input_tokenized_data["tokens"].clone()
                for f_coord, s_coord in current_candidate_mutations:
                    trial_tokens[optim_mask[f_coord]] = best_tokens_indices[(f_coord, s_coord)]

                if substitution_validity_function is not None:
                    if not substitution_validity_function(trial_tokens, tokenizer=tokenizer, masks_data=masks_data):
                        all_substitutions_valid = False
                        break
            
            if all_substitutions_valid:
                indices_to_sample_batches.append(current_candidate_mutations)

        # 4. Build the final candidate tensors
        candidates_list = []
        for input_tokenized_data in input_tokenized_data_list:
            input_new_candidates = []
            masks_data = input_tokenized_data["masks"]
            optim_mask = masks_data["optim_mask"]
            
            for mutation_bundle in indices_to_sample_batches:
                new_candidate = input_tokenized_data["tokens"].clone()
                for f_coord, s_coord in mutation_bundle:
                    new_candidate[optim_mask[f_coord]] = best_tokens_indices[(f_coord, s_coord)]
                input_new_candidates.append(new_candidate)
                
            candidates_list.append(torch.stack(input_new_candidates))
        
        # Increment step for the next call
        self.current_step += 1
        return candidates_list