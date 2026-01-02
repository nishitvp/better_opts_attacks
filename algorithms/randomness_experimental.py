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
    def __init__(self, mutation_schedule=None):
        """
        Args:
            mutation_schedule: list of (num_steps, mutation_count) tuples.
                Example:
                    [(200, 8), (200, 6), (200, 4)]
                After the schedule is exhausted, mutation_count defaults to 1.
        """
        self.mutation_schedule = mutation_schedule or []
        self.current_step = 0

        # Precompute cumulative step boundaries for fast lookup
        self._schedule_boundaries = []
        total = 0
        for steps, mutations in self.mutation_schedule:
            total += steps
            self._schedule_boundaries.append((total, mutations))

    def _get_mutation_count(self):
        """Return mutation count for the current step."""
        for boundary, mutations in self._schedule_boundaries:
            if self.current_step < boundary:
                return mutations
        return 1  # default fallback

    def __call__(
        self,
        tokenizer,
        best_tokens_indices,
        input_tokenized_data_list,
        substitution_validity_function,
        max_candidate_size,
    ):
        # 1. Determine mutation count from schedule
        mutation_count = self._get_mutation_count()

        indices_to_sample_batches = []  # List[Set[(first_coord, second_coord)]]

        # 2. Candidate Generation Loop
        while len(indices_to_sample_batches) < max_candidate_size:
            current_candidate_mutations = set()

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

                trial_tokens = input_tokenized_data["tokens"].clone()
                for f_coord, s_coord in current_candidate_mutations:
                    trial_tokens[optim_mask[f_coord]] = best_tokens_indices[(f_coord, s_coord)]

                if substitution_validity_function is not None:
                    if not substitution_validity_function(
                        trial_tokens,
                        tokenizer=tokenizer,
                        masks_data=masks_data,
                    ):
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

        self.current_step += 1
        return candidates_list
