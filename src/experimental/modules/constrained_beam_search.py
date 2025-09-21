from typing import Optional, Tuple, Any, Dict
import torch
import pdb
class ConstrainedBeamSearch:
    """Constrained beam search utility class for discrete diffusion models."""
    
    def __init__(self, diffusion_module):
        """Initialize with a reference to the parent diffusion module."""
        self.diffusion_module = diffusion_module

    def beam_search_generation(
        self,
        model_input: torch.Tensor,
        label_data: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Main constrained beam search generation function."""
        num_candidates = self.diffusion_module.diffusion_config['num_candidates']
        labels = label_data.labels['sequence_data']
        label_locations = label_data.label_location['sequence_data']

        transformed_masked_input, transformed_labels = self.diffusion_module.transform_input_and_labels(
            model_input.transformed_sequences['sequence_data'],
            label_data.labels['sequence_data'], 
            label_data.label_location['sequence_data']
        )

        # NOTE: This is same for all batches, so we can generate it once
        topk_neighbors, topk_similarity = self.generate_semantic_graph(num_candidates)

        if self.diffusion_module.diffusion_config['constrained_beam_initial'] == 'generated':
            cur_beam_candidates = self.get_generated_initial_candidates(
                transformed_masked_input, transformed_labels, 
                label_data.label_location['sequence_data'], num_candidates
            )
        elif self.diffusion_module.diffusion_config['constrained_beam_initial'] == 'random':
            cur_beam_candidates = self.get_random_initial_candidates(
                transformed_masked_input, transformed_labels, 
                label_data.label_location['sequence_data'], num_candidates
            )

        for i in range(self.diffusion_module.diffusion_config['constrained_beam_refinement_steps']):
            # Expand candidates based on topk neighbors
            expanded_candidates, expanded_values = self.get_expanded_candidates_graphs(
                cur_beam_candidates, transformed_labels, label_data.label_location['sequence_data'], 
                topk_neighbors, topk_similarity, num_candidates
            )

            # Prune top num_candidates
            cur_beam_candidates, cur_beam_vals = self.prune_candidates(
                expanded_candidates, expanded_values, transformed_labels, 
                label_data.label_location['sequence_data'], num_candidates
            )

        bs, seq_len = transformed_masked_input.shape
        batch_indices, seq_indices = label_data.label_location['sequence_data'][:, 0], label_data.label_location['sequence_data'][:, 1]
        generated_ids = torch.zeros((bs, num_candidates, self.diffusion_module.num_hierarchies), dtype=torch.long, device=self.diffusion_module.device)
        
        for i in range(num_candidates):
            cur_seq = cur_beam_candidates[:, i, :]
            generated_ids[:, i, :] = cur_seq[batch_indices, seq_indices].reshape(-1, self.diffusion_module.num_hierarchies)

        return generated_ids, cur_beam_vals

    def get_generated_initial_candidates(
        self, 
        masked_input: torch.Tensor, 
        labels: torch.Tensor, 
        label_locations: torch.Tensor, 
        num_candidates: int
    ) -> torch.Tensor:
        """Generate initial candidates using model predictions."""
        bs, seq_len = masked_input.shape

        encoder_output = self.diffusion_module.forward(masked_input)
        probabilities = self.diffusion_module.encoder_output_to_probabilities(encoder_output)

        if self.diffusion_module.diffusion_config.get('maskout_masking_token_prob_decoding', False):
            # Set probabilities of masking token to zero
            probabilities[:, :, self.diffusion_module.masking_token_id] = 0.0
        
        seq_ids, pos_ids = label_locations[:, 0], label_locations[:, 1]
        probs_at_locations = probabilities[seq_ids, pos_ids].reshape(bs, self.diffusion_module.num_hierarchies, -1)
        topk_probs, topk_indices = torch.topk(probs_at_locations, num_candidates // 3, dim=-1)

        cur_beam_candidates = torch.zeros((bs, num_candidates, seq_len), dtype=torch.long, device=self.diffusion_module.device)

        cur_beam_candidates[:, :, :] = masked_input.unsqueeze(1).expand(-1, num_candidates, -1)
        cur_beam_candidates[seq_ids, 0, pos_ids] = probs_at_locations.argmax(dim=-1).flatten()

        for i in range(1, num_candidates):
            for h in range(self.diffusion_module.num_hierarchies):
                # For each hierarchy level, sample from the topk candidates
                probs_h = topk_probs[:, h, :]  # Shape: (bs, candidates)
                indices_h = topk_indices[:, h, :]  # Shape: (bs, candidates)

                # Sample one index per batch for this hierarchy
                sampled_idx = torch.multinomial(probs_h, 1).squeeze(-1)  # Shape: (bs,)
                cur_beam_candidates[torch.arange(bs), i, pos_ids[h::self.diffusion_module.num_hierarchies]] = indices_h[torch.arange(bs), sampled_idx]

        return cur_beam_candidates

    def get_random_initial_candidates(
        self, 
        masked_input: torch.Tensor, 
        labels: torch.Tensor, 
        label_locations: torch.Tensor, 
        num_candidates: int
    ) -> torch.Tensor:
        """Generate initial candidates using random sampling from codebooks."""
        if self.diffusion_module.transformed_codebooks is None:
            self.diffusion_module.transformed_codebooks, _ = self.diffusion_module.transform_input_and_labels(
                self.diffusion_module.codebooks.to(self.diffusion_module.device), None, None
            )

        total_items = self.diffusion_module.transformed_codebooks.shape[0]

        bs, seq_len = masked_input.shape
        random_indices = torch.randint(0, total_items, (masked_input.shape[0] * num_candidates,), device=self.diffusion_module.device)
        cur_masked_vals = self.diffusion_module.transformed_codebooks[random_indices].reshape(
            masked_input.shape[0], num_candidates, -1
        )

        cur_beam_candidates = torch.zeros((bs, num_candidates, seq_len), dtype=torch.long, device=self.diffusion_module.device)
        cur_beam_candidates[:, :, :] = masked_input.unsqueeze(1).expand(-1, num_candidates, -1)
        seq_ids, pos_ids = label_locations[:, 0], label_locations[:, 1]

        for i in range(num_candidates):
            cur_beam_candidates[seq_ids, i, pos_ids] = cur_masked_vals[:, i, :].flatten()

        return cur_beam_candidates

    def find_matching_indices(self, cur_masked_vals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find matching indices in the codebook for given values."""
        if self.diffusion_module.transformed_codebooks is None:
            self.diffusion_module.transformed_codebooks, _ = self.diffusion_module.transform_input_and_labels(
                self.diffusion_module.codebooks.to(self.diffusion_module.device), None, None
            )
        
        cur_expanded = cur_masked_vals.unsqueeze(1)  # [bs, 1, num_hierarchies]
        codebooks_expanded = self.diffusion_module.transformed_codebooks.unsqueeze(0)  # [1, num_items, num_hierarchies]

        # Check equality across all hierarchies: [bs, num_items, num_hierarchies] -> [bs, num_items]
        matches = (cur_expanded == codebooks_expanded).sum(dim=2)
        inds = matches.argmax(dim=1)  # [bs]
        
        return inds, matches

    def get_expanded_candidates_graphs(
        self, 
        cur_beam_candidates: torch.Tensor, 
        labels: torch.Tensor, 
        label_locations: torch.Tensor,
        topk_neighbors: torch.Tensor, 
        topk_similarity: torch.Tensor, 
        num_candidates: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Expand candidates based on semantic graph neighbors."""
        bs, cur_beam_size, seq_len = cur_beam_candidates.shape
        expanded_candidates = torch.zeros((bs, num_candidates * cur_beam_size, seq_len), dtype=torch.long, device=self.diffusion_module.device)
        expanded_values = torch.zeros((bs, num_candidates * cur_beam_size), dtype=torch.float32, device=self.diffusion_module.device)

        batch_ids, seq_ids = label_locations[:, 0], label_locations[:, 1]
        for i in range(cur_beam_size):
            cur_seq = cur_beam_candidates[:, i, :].clone()
            cur_masked_vals = cur_seq[batch_ids, seq_ids].reshape(-1, self.diffusion_module.num_hierarchies)

            predicted_items, matches = self.find_matching_indices(cur_masked_vals)
            expanded_inds = topk_neighbors[predicted_items]
            extracted_vals = topk_similarity[predicted_items]

            for j in range(num_candidates):
                new_masked_vals = self.diffusion_module.transformed_codebooks[expanded_inds[:, j], :]

                expanded_candidates[:, i * num_candidates + j, :] = cur_seq.clone()
                expanded_candidates[batch_ids, i * num_candidates + j, seq_ids] = new_masked_vals.flatten()
                expanded_values[:, i * num_candidates + j] = extracted_vals[:, j]
        
        return expanded_candidates, expanded_values

    def one_step_generation_value(self, cur_expanded_candidates: torch.Tensor, label_locations: torch.Tensor) -> torch.Tensor:
        """Generate values for one step."""
        seq_ids, pos_ids = label_locations[:, 0], label_locations[:, 1]
            
        masked_input = cur_expanded_candidates.clone()
        masked_input[label_locations[:, 0], label_locations[:, 1]] = self.diffusion_module.masking_token_id
        encoder_output = self.diffusion_module.forward(masked_input)
        probabilities = self.diffusion_module.encoder_output_to_probabilities(encoder_output)

        # Calculate the value for this candidate
        masked_vals = cur_expanded_candidates[seq_ids, i, pos_ids]
        probs_at_locations = probabilities[seq_ids, pos_ids]
        correct_probs = probs_at_locations[torch.arange(bs * self.diffusion_module.num_hierarchies), masked_vals]
        
        return correct_probs.reshape(-1, self.diffusion_module.num_hierarchies).prod(dim=-1)

    def multi_step_generation_value(self, cur_expanded_candidates: torch.Tensor, label_locations: torch.Tensor) -> torch.Tensor:
        """Generate values for multi-step generation."""
        bs, seq_len = cur_expanded_candidates.shape
        generated_values = torch.ones((bs,), dtype=torch.float32, device=self.diffusion_module.device)
        
        cur_masked_input = cur_expanded_candidates.clone()
        cur_masked_input[label_locations[:, 0], label_locations[:, 1]] = self.diffusion_module.masking_token_id
        for i in range(self.diffusion_module.num_hierarchies):
            encoder_output = self.diffusion_module.forward(cur_masked_input)
            probabilities = self.diffusion_module.encoder_output_to_probabilities(encoder_output)
            
            cur_mask = (cur_masked_input == self.diffusion_module.masking_token_id)
            cur_masked_vals = cur_expanded_candidates[cur_mask]
            cur_mask_batch, cur_mask_seq = torch.where(cur_mask)
            probs_at_locations = probabilities[cur_mask]

            transfer_positions = torch.zeros_like(cur_mask, dtype=torch.bool, device=self.diffusion_module.device)
            if self.diffusion_module.diffusion_config['unmasking_type'] == 'top-prob':
                top_values = probs_at_locations.max(dim=-1).values.reshape(bs, -1)
                logits = top_values / self.diffusion_module.diffusion_config['unmasking_temperature']  # (bs, num_mask_per_row)
                probs = torch.softmax(logits, dim=1)
                random_offsets = torch.multinomial(probs, 1)

            num_mask_per_seq = self.diffusion_module.num_hierarchies - i
            
            selected_indices = torch.arange(bs, device=self.diffusion_module.device)[:, None] * num_mask_per_seq + random_offsets
            selected_indices = selected_indices.flatten()
            selected_batch_indices = cur_mask_batch[selected_indices]
            selected_seq_indices = cur_mask_seq[selected_indices]

            transfer_positions[selected_batch_indices, selected_seq_indices] = True
            transfer_index_t_s = transfer_positions[cur_mask]
            probs_at_transfer = probabilities[cur_mask][transfer_index_t_s]
            try:
                generated_values = generated_values * probs_at_transfer[torch.arange(bs), cur_masked_vals[transfer_index_t_s]]
            except:
                pdb.set_trace()
            cur_masked_input[selected_batch_indices, selected_seq_indices] = cur_masked_vals[transfer_index_t_s]
        
        return generated_values

    def prune_candidates(
        self, 
        expanded_candidates: torch.Tensor, 
        expanded_values: torch.Tensor, 
        labels: torch.Tensor, 
        label_locations: torch.Tensor, 
        num_candidates: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prune expanded candidates to keep only the top num_candidates."""
        bs, num_candidates_expanded, seq_len = expanded_candidates.shape

        generated_values = torch.zeros((bs, num_candidates_expanded), dtype=torch.float32, device=self.diffusion_module.device)
        for i in range(num_candidates_expanded):
            if self.diffusion_module.diffusion_config['prune_generation_values'] == 'one-step':
                generated_values[:, i] = self.one_step_generation_value(expanded_candidates[:, i, :].clone(), label_locations)
            elif self.diffusion_module.diffusion_config['prune_generation_values'] == 'multi-step':
                generated_values[:, i] = self.multi_step_generation_value(expanded_candidates[:, i, :].clone(), label_locations)

        removed_generated_values = self.zero_out_duplicate_candidate_values(
            expanded_candidates.reshape(-1, seq_len), generated_values.flatten()
        )
        topk_vals, topk_inds = torch.topk(removed_generated_values.reshape(-1, num_candidates_expanded), num_candidates, dim=-1)
        cur_beam_candidates = torch.zeros((bs, num_candidates, seq_len), dtype=torch.long, device=self.diffusion_module.device)

        for i in range(num_candidates):
            cur_beam_candidates[:, i, :] = expanded_candidates[torch.arange(bs), topk_inds[:, i]]

        return cur_beam_candidates, topk_vals

    
    def zero_out_duplicate_candidate_values(
        self, 
        expanded_candidates_reshaped: torch.Tensor, 
        expanded_values_reshaped: torch.Tensor
    ) -> torch.Tensor:
        """Zero out values for duplicate candidates to avoid selecting the same candidate multiple times."""
        modified_values = expanded_values_reshaped.clone()
        N, seq_len = expanded_candidates_reshaped.shape
        
        # Compute pairwise equality matrix
        # This creates a boolean tensor where entry (i,j) is True if row i equals row j
        candidates_expanded_i = expanded_candidates_reshaped.unsqueeze(1)  # (N, 1, seq_len)
        candidates_expanded_j = expanded_candidates_reshaped.unsqueeze(0)  # (1, N, seq_len)
        
        # Check equality across all elements in each row pair
        row_equality = (candidates_expanded_i == candidates_expanded_j).all(dim=2)  # (N, N)
        
        indices = torch.arange(N, device=expanded_candidates_reshaped.device)
        indices_i = indices.unsqueeze(1)  # (N, 1)
        indices_j = indices.unsqueeze(0)  # (1, N)
        
        # A row is a duplicate if there exists an identical row with a smaller index
        has_earlier_duplicate = (row_equality & (indices_j < indices_i)).any(dim=1)
        
        modified_values[has_earlier_duplicate] = 0.0

        return modified_values

    def generate_semantic_graph(self, num_candidates: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate semantic graph based on similarity between codebook embeddings."""
        if self.diffusion_module.transformed_codebooks is None:
            self.diffusion_module.transformed_codebooks, _ = self.diffusion_module.transform_input_and_labels(
                self.diffusion_module.codebooks.to(self.diffusion_module.device), None, None
            )

        transformed_codebooks_embeddings = self.diffusion_module.item_sid_embedding_table_encoder(self.diffusion_module.transformed_codebooks)
        emb_flat = transformed_codebooks_embeddings.reshape(self.diffusion_module.transformed_codebooks.shape[0], -1)
        similarity = torch.matmul(emb_flat, emb_flat.t())

        topk_similarity, topk_neighbors = torch.topk(similarity, num_candidates, dim=1)
        return topk_neighbors, topk_similarity
    
    def zero_out_duplicate_candidate_values(self, expanded_candidates_reshaped: torch.Tensor, expanded_values_reshaped: torch.Tensor) -> torch.Tensor:
        modified_values = expanded_values_reshaped.clone()
        N, seq_len = expanded_candidates_reshaped.shape
        
        # Compute pairwise equality matrix
        # This creates a boolean tensor where entry (i,j) is True if row i equals row j
        candidates_expanded_i = expanded_candidates_reshaped.unsqueeze(1)  # (N, 1, seq_len)
        candidates_expanded_j = expanded_candidates_reshaped.unsqueeze(0)  # (1, N, seq_len)
        
        # Check equality across all elements in each row pair
        row_equality = (candidates_expanded_i == candidates_expanded_j).all(dim=2)  # (N, N)
        
        indices = torch.arange(N, device=expanded_candidates_reshaped.device)
        indices_i = indices.unsqueeze(1)  # (N, 1)
        indices_j = indices.unsqueeze(0)  # (1, N)
        
        # A row is a duplicate if there exists an identical row with a smaller index
        has_earlier_duplicate = (row_equality & (indices_j < indices_i)).any(dim=1)
        
        modified_values[has_earlier_duplicate] = 0.0

        return modified_values