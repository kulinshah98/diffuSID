from typing import Optional, Tuple, Any, Dict

import numpy as np
import torch
import torch.nn as nn
import pdb
import math
import pickle
import gcsfs

from src.models.modules.base_module import BaseModule
from src.models.modules.semantic_id.tiger_generation_model import SemanticIDEncoderModule
from src.utils.utils import (
    delete_module,
    find_module_shape,
    get_parent_module_and_attr,
    reset_parameters,
)
from src.data.loading.components.interfaces import (
    SequentialModelInputData,
    SequentialModuleLabelData,
)
from src.experimental.modules.constrained_beam_search import ConstrainedBeamSearch
from src.experimental.modules.rotary_position_encoding import (
    RotaryTransformerEncoder,
    RotaryTransformerEncoderLayer
)
from itertools import product




class DiscreteDiffusionModule(BaseModule):
    """Module for discrete diffusion models."""

    def __init__(
        self, 
        model: torch.nn.Module,
        use_token_type_embedding: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_function: torch.nn.Module,
        evaluator: Optional[object],
        num_hierarchies: int,
        vocab_size: int,
        embedding_dim: int,
        padding_token_id: int,
        normalization: Optional[bool] = True,
        use_dense_retrieval_head: Optional[bool] = False,
        dense_retrieval_head: Optional[torch.nn.Module] = None,
        dense_retrieval_loss_function: Optional[torch.nn.Module] = None,
        dense_retrieval_update_frequency: Optional[int] = 5,
        positional_embedding: Optional[torch.nn.Module] = None,
        use_diff_semantic_id: Optional[bool] = False,
        ignore_diff_semid_padloss: Optional[bool] = False,
        mask_diff_semid_pad_attn: Optional[bool] = False,
        attend_to_padding: Optional[bool] = True,
        token_type_embedding: Optional[torch.nn.Module] = None,
        data_freqs_path: Optional[str] = None,
        projection: Optional[bool] = True,
        deduplication: Optional[bool] = False,
        codebooks: Optional[torch.Tensor] = None,
        diffusion_config: Optional[dict] = None,
        training_loop_function: Optional[callable] = None,
        use_rotary_position_encoding: Optional[bool] = False,
        max_position_embeddings: Optional[int] = 2048,
        eval_hierarchy_cutoff: Optional[int] = 1,
        update_frequency: Optional[int] = 1000,
        **kwargs,
    ) -> None:

        super().__init__(
            model=model,
            optimizer=optimizer, 
            scheduler=scheduler,
            loss_function=loss_function,
            evaluator=evaluator,
            training_loop_function=training_loop_function
        )
        
        self.positional_embedding = positional_embedding
        self.use_token_type_embedding = use_token_type_embedding
        self.use_dense_retrieval_head = use_dense_retrieval_head
        self.dense_retrieval_head = dense_retrieval_head
        self.dense_retrieval_loss_function = dense_retrieval_loss_function
        self.dense_retrieval_update_frequency = dense_retrieval_update_frequency
        if self.use_token_type_embedding:
            self.token_type_embedding = token_type_embedding

        self.use_rotary_position_encoding = use_rotary_position_encoding

        # Initialize additional parameters for discrete diffusion
        self.transformed_codebooks = None
        self.diffusion_config = diffusion_config
        self.num_hierarchies = num_hierarchies
        self.deduplication = deduplication
        self.projection = projection
        self.attend_to_padding = attend_to_padding
        self.eval_hierarchy_cutoff = eval_hierarchy_cutoff
        self.normalization = normalization
        self.use_diff_semantic_id = use_diff_semantic_id
        self.ignore_diff_semid_padloss = ignore_diff_semid_padloss
        self.mask_diff_semid_pad_attn = mask_diff_semid_pad_attn

        self.update_frequency = update_frequency

        self.num_embeddings_per_hierarchy = vocab_size + 1  # +1 for masking
        if self.use_diff_semantic_id:
            self.num_embeddings_per_hierarchy += 1  # +1 for extra token for augmentation
        self.embedding_dim = embedding_dim

        fs = gcsfs.GCSFileSystem()
        self.freqs = None
        self.sorted_freqs = None
        if data_freqs_path is not None:
            with fs.open(data_freqs_path, "rb") as f:
                data = pickle.load(f)

            self.items = torch.Tensor(data["items"])
            self.freqs = torch.Tensor(data["freqs"])
            self.sorted_freqs = self.freqs.clone()
            self.sorted_freqs = self.sorted_freqs.sort().values
        
        self.masking_token_id = vocab_size
        self.padding_token_id = padding_token_id
        self.saved_batch = None
        self.vocab_size = vocab_size
        
        self.codebooks = codebooks.t()
        assert (
            self.codebooks.size(1) == num_hierarchies
        ), self.codebooks.shape #"codebooks should be of shape (-1, num_hierarchies)"

        # generate embedding tables for each hierarchy
        # here we assume each hierarchy has the same amount of embeddings

        ## TODO (Kulin): There are redundant masking embeddings which are not getting used
        # Initialize the item semantic ID embedding table internally
        self.item_sid_embedding_table_encoder = torch.nn.Embedding(
            num_embeddings=self.num_embeddings_per_hierarchy * self.num_hierarchies + 1,  # +1 for padding token
            embedding_dim=embedding_dim
        )

    def encoder_output_to_loss(
        self,
        encoder_output: torch.Tensor,
        labels: torch.Tensor,
        label_locations: torch.Tensor,
    ):
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        query_embeddings = encoder_output[
            label_locations[:, 0], label_locations[:, 1]
        ]
        key_embeddings = self.item_sid_embedding_table_encoder.weight

        if self.loss_function.normalize:
            query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=-1)
            key_embeddings = torch.nn.functional.normalize(key_embeddings, dim=-1)
        
        logits = torch.mm(query_embeddings, key_embeddings.t())
        loss = loss_fn(logits, labels)
        return loss

    def forward(
        self,
        input: SequentialModelInputData):
        """ Forward pass for the discrete diffusion model.
        """
        bs, seq_len = input.shape
        encoded_input = self.item_sid_embedding_table_encoder(input) # shape: (batch_size, sequence_length, d_model)
        
        if self.use_rotary_position_encoding:
            # Use rotary position encoding - no absolute position embeddings added to input
            # RoPE is applied within the attention mechanism itself
            inputs_emb = encoded_input
        else:
            position_ids = torch.arange(seq_len, device=input.device).unsqueeze(0).expand(bs, -1)
            pos_embeddings = self.positional_embedding(position_ids)
            
            if self.use_token_type_embedding:
                token_type_ids = position_ids % self.num_hierarchies  # Shape: (bs, seq_len)
                token_type_embeddings = self.token_type_embedding(token_type_ids)
                pos_embeddings += token_type_embeddings
            
            inputs_emb = encoded_input + pos_embeddings
        
        padding_mask = None
        if not self.attend_to_padding:
            new_padding_value = self.num_embeddings_per_hierarchy * self.num_hierarchies
            padding_mask = (input == new_padding_value)
        
        if self.use_diff_semantic_id and self.mask_diff_semid_pad_attn:
            origin_input = input % self.num_embeddings_per_hierarchy
            diff_semantic_id_padding_mask = (origin_input == self.num_embeddings_per_hierarchy - 1)
            padding_mask = padding_mask | diff_semantic_id_padding_mask if padding_mask is not None else diff_semantic_id_padding_mask
        
        output = self.model(inputs_emb, src_key_padding_mask=padding_mask)
        
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(f"Parameter '{name}' did not receive a gradient.")

        # pdb.set_trace()
        if self.normalization:
            return output / math.sqrt(self.embedding_dim)
        else:   
            return output


    def model_step(
        self,
        model_input: torch.Tensor,
        label_data: torch.Tensor,
        fractions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the discrete diffusion model.
        """
        masked_input = model_input.transformed_sequences['sequence_data']
        labels = label_data.labels['sequence_data']
        label_locations = label_data.label_location['sequence_data']

        # Transform both input and labels with hierarchical offsets
        transformed_masked_input, transformed_labels = self.transform_input_and_labels(
            masked_input, labels, label_locations
        )
        masked_locations = (transformed_masked_input == self.masking_token_id)

        encoder_output = self.forward(transformed_masked_input)

        if self.diffusion_config['loss_reweighting'] == 'equal':
            weights = torch.ones_like(transformed_labels, dtype=torch.float32)
        elif self.diffusion_config['loss_reweighting'] == 'normalized':
            weights = 1.0 / fractions[ torch.where(masked_locations)[0] ]

        if self.use_diff_semantic_id and self.ignore_diff_semid_padloss:
            original_ids = transformed_labels % self.num_embeddings_per_hierarchy
            diff_semantic_id_padding_token = ~(original_ids == self.num_embeddings_per_hierarchy - 1)
            weights = weights * diff_semantic_id_padding_token.float()

        loss = self.loss_function(
            query_embeddings=encoder_output, 
            key_embeddings=self.item_sid_embedding_table_encoder.weight,
            label_locations=label_locations,
            labels=transformed_labels,
            weights=weights,
            )
        
        avg_loss = loss / torch.sum(weights)
        
        # Always use dense_retrieval_head to avoid DDP unused parameter issues
        if self.use_dense_retrieval_head and self.dense_retrieval_head is not None:
            bs, seq_len = masked_input.shape

            self.transformed_codebooks, _ = self.transform_input_and_labels(
                self.codebooks.to(self.device), None, None
            )
            all_embeddings = self.item_sid_embedding_table_encoder.weight[self.transformed_codebooks]
            all_embeddings_reshaped = all_embeddings.view(self.codebooks.shape[0], -1)
            projected_all_embeddings = self.dense_retrieval_head(all_embeddings_reshaped)

            # Only compute and add dense retrieval loss at specified frequency
            if self.trainer.global_step % self.dense_retrieval_update_frequency == 0:
                generated_encodings = encoder_output[ label_locations[:, 0], label_locations[:, 1] ]
                generated_encodings_reshaped = generated_encodings.view(bs, -1)
                
                curent_labels = transformed_labels.view(bs, 1, -1)
                matches = (curent_labels == self.transformed_codebooks.unsqueeze(0)).all(dim=2)
                label_items = torch.where(matches)[1]

                dense_retrieval_loss = self.dense_retrieval_loss_function(
                    query_embeddings=generated_encodings_reshaped, 
                    key_embeddings=projected_all_embeddings,
                    labels=label_items,
                )
                avg_loss += dense_retrieval_loss
            else:
                # Use a dummy loss to ensure the dense_retrieval_head parameters are used
                # This prevents DDP from complaining about unused parameters
                dummy_loss = torch.tensor(0.0, device=avg_loss.device, requires_grad=True)
                # Create a small computation that uses the projected embeddings
                dummy_loss = dummy_loss + (projected_all_embeddings.mean() * 0.0)
                avg_loss = avg_loss + dummy_loss
        
        return encoder_output, avg_loss

    def encoder_output_to_probabilities(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Convert encoder output to probabilities using dot product with embedding weights.
        """        
        # (bs, seq_len, d_model) @ (d_model, vocab_size) = (bs, seq_len, vocab_size)
        key_embeddings = self.item_sid_embedding_table_encoder.weight
        if self.loss_function.normalize:
            encoder_output = torch.nn.functional.normalize(encoder_output, dim=-1)
            key_embeddings = torch.nn.functional.normalize(key_embeddings, dim=-1)

        logits = torch.matmul(encoder_output, key_embeddings.t()) / self.loss_function.tau
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

    def find_row_with_sequence(self, target_sequence: torch.Tensor) -> torch.Tensor:
        """
        Find which row in self.codebooks contains the target sequence.
        Args: target_sequence: Tensor of shape [4] containing the four numbers to search for
        Returns: Tensor containing the row indices where the sequence is found
        """
        if target_sequence.dim() == 1:
            target_sequence = target_sequence.unsqueeze(0)  # [1, 4]
        target_sequence = target_sequence.to(self.codebooks.device)
        
        # Compare each row with the target sequence
        # This creates a boolean tensor of shape [24202, 4]
        matches = (self.codebooks == target_sequence)
        
        # Find rows where all 4 elements match
        # all(dim=1) checks if all elements in each row are True
        row_matches = matches.all(dim=1)  # [24202]
        
        # Get the indices of matching rows
        matching_indices = torch.where(row_matches)[0]
        
        return matching_indices

    def process_masked_input(self, masked_input: torch.Tensor) -> torch.Tensor:
        """
        Process masked input by replacing padding tokens (-1) with new calculated padding value.
        """
        new_padding_value = self.num_embeddings_per_hierarchy * self.num_hierarchies
        
        processed_input = masked_input.clone()
        processed_input[masked_input == self.padding_token_id] = new_padding_value
        
        return processed_input

    def get_modified_eval_inputs(
        self,
        masked_input: torch.Tensor,
        label_locations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate modified masked input and label locations for evaluation steps.

        This function expands label locations to cover self.num_hierarchies consecutive positions
        (corresponding to the self.num_hierarchies hierarchies) and sets masking tokens at those positions.

        """
        modified_masked_input = masked_input.clone()
        row_indices = label_locations[::self.num_hierarchies, 0]  # Shape: [N] - batch indices
        col_indices = label_locations[::self.num_hierarchies, 1]  # Shape: [N] - sequence positions

        # Create four consecutive column positions for each location
        # This corresponds to the 4 hierarchical levels
        offsets = torch.arange(self.num_hierarchies, device=self.device)  # [0, 1, 2, 3, .., num_hierarchies-1]
        new_col_indices = col_indices.unsqueeze(1) + offsets.unsqueeze(0)  # [N, num_hierarchies]
        # Repeat row indices to match the expanded columns
        new_row_indices = row_indices.unsqueeze(1).repeat(1, self.num_hierarchies)  # [N, num_hierarchies]

        # Flatten to get new label_locations [N*num_hierarchies, 2]
        modified_label_locations = torch.stack([
            new_row_indices.flatten(),
            new_col_indices.flatten()
        ], dim=1)
        
        # Set masking tokens at all new locations in the modified input
        modified_masked_input[new_row_indices.flatten(), new_col_indices.flatten()] = self.masking_token_id
        
        return modified_masked_input, modified_label_locations

    def training_step(
        self,
        batch: Tuple[Tuple[SequentialModelInputData, SequentialModuleLabelData]],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.
        """
        
        # # Update global training step for dynamic collate function (fallback for non-multiprocessing)
        # from src.data.loading.components.dynamic_collate_functions import set_training_step
        # set_training_step(self.trainer.global_step)
        
        batch = batch[0]
        model_input: SequentialModelInputData = batch[0]
        label_data: SequentialModuleLabelData = batch[1]
        
        # Apply masking to input and labels
        model_input, label_data, fractions = self.mask_input_and_labels(model_input, label_data)
        model_output, loss = self.model_step(model_input, label_data, fractions)

        # Access the actual optimizer instance using self.optimizers()
        optimizer = self.optimizers()
        if hasattr(optimizer, 'param_groups'):
            current_lr = optimizer.param_groups[0]['lr']
        else:
            # Handle case where optimizer might be a list
            current_lr = optimizer[0].param_groups[0]['lr'] if isinstance(optimizer, list) else None
        
        if current_lr is not None and batch_idx % self.update_frequency == 0:
            self.log("learning_rate", current_lr, on_step=True, on_epoch=False, prog_bar=True)

        if self.training_loop_function is not None:
            self.training_loop_function(self, loss)

        return loss

    def convert_deduplicate(
        self,
        generated_ids: torch.Tensor,
        transformed_labels: torch.Tensor,
        final_probs: torch.Tensor,
        model_input: SequentialModelInputData,
        label_data: SequentialModuleLabelData,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.transformed_codebooks is None:
            self.transformed_codebooks, _ = self.transform_input_and_labels(
                self.codebooks.to(self.device), None, None
            )
        bs, num_candidates, _ = generated_ids.shape

        deduplicated_generated_ids = torch.zeros(
            (bs, num_candidates, self.num_hierarchies + 1), 
            dtype=torch.long, device=self.device
        )
        modified_final_probs = torch.zeros(
            (bs, num_candidates), dtype=torch.float, device=self.device
        )
        for i in range(bs):
            cur_start = 0
            remaining = num_candidates
            for j in range(num_candidates):
                # Get the generated sequence for this candidate
                generated_sequence = generated_ids[i, j, :]

                # Find the row in codebooks that matches the generated sequence
                num_duplicates = (self.transformed_codebooks == generated_sequence).all(dim=1).sum()
                deduplication_ids = torch.randperm(num_duplicates, device=self.device)[:min(remaining, num_duplicates)]

                cur_end = cur_start + len(deduplication_ids)
                deduplicated_generated_ids[i, cur_start:cur_end, :self.num_hierarchies] = generated_sequence  # Default to original sequence
                deduplicated_generated_ids[i, cur_start:cur_end, self.num_hierarchies] = deduplication_ids
                modified_final_probs[i, cur_start:cur_end] = final_probs[i, j]  # Use the original probability

                remaining -= len(deduplication_ids)
                cur_start = cur_end

                if remaining <= 0:
                    break
                
        # deduplicated_generated_ids[:, :, :self.num_hierarchies] = generated_ids
        # deduplicated_generated_ids[:, :, self.num_hierarchies] = 0
        # Convert labels to the same format
        deduplicated_generated_labels = torch.zeros(
            (bs, self.num_hierarchies + 1), 
            dtype=torch.long, device=self.device
        )
        transformed_labels_reshaped = transformed_labels.reshape(bs, self.num_hierarchies)
        for i in range(bs):
            # Get the label sequence for this candidate
            label_sequence = transformed_labels_reshaped[i, :]

            num_duplicates = (self.transformed_codebooks == label_sequence).all(dim=1).sum()
            deduplication_ids = torch.randperm(num_duplicates, device=self.device)[:1]

            deduplicated_generated_labels[i, :self.num_hierarchies] = label_sequence
            deduplicated_generated_labels[i, self.num_hierarchies] = deduplication_ids

        return deduplicated_generated_ids, deduplicated_generated_labels.flatten(), modified_final_probs

    def project_generated_ids(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """
        For each generated_id, if it exists in transformed_codebooks, keep it as is.
        Otherwise, replace it with the codebook entry with maximum similarity (dot product).
        Args:
            generated_ids: Tensor of shape (bs, num_candidates, num_hierarchies)
        Returns:
            projected_ids: Tensor of same shape as generated_ids
        """
        if self.transformed_codebooks is None:
            self.transformed_codebooks, _ = self.transform_input_and_labels(
                self.codebooks.to(self.device), None, None
            )
        bs, num_candidates, num_hierarchies = generated_ids.shape
        codebooks = self.transformed_codebooks  # (num_codebooks, num_hierarchies)
        # Reshape for vectorized comparison
        flat_gen = generated_ids.view(-1, num_hierarchies)  # (bs*num_candidates, num_hierarchies)
        codebooks_exp = codebooks.unsqueeze(0)  # (1, num_codebooks, num_hierarchies)
        flat_gen_exp = flat_gen.unsqueeze(1)    # (bs*num_candidates, 1, num_hierarchies)
        # Check for exact match
        matches = (flat_gen_exp == codebooks_exp).sum(dim=2)  # (bs*num_candidates, num_codebooks)
        match_idx = matches.float().argmax(dim=1)  # (bs*num_candidates,)
        projected_ids_flat = codebooks[match_idx]
        return projected_ids_flat.reshape(bs, num_candidates, num_hierarchies)

    def generative_retrieval_eval(
        self, generated_ids, masked_input, labels, label_locations, 
        modified_masked_input, modified_label_locations, 
        transformed_labels, final_probs, model_input, label_data
        ):
        
        
        if self.freqs is not None:
            bs = generated_ids.shape[0]
            labels_reshaped = labels.view(bs, self.num_hierarchies)
            matches = (labels_reshaped.unsqueeze(1) == self.codebooks.to(self.device).unsqueeze(0))
            row_matches = matches.all(dim=2)
            label_indices = row_matches.float().argmax(dim=1)
            label_freqs = self.freqs.to(self.device).clone().detach()[label_indices.to(self.device)]

        first_sid_label_locations = label_locations[0::self.num_hierarchies, 1].clone().detach()
        # import ipdb; ipdb.set_trace()
        self.evaluator(
            marginal_probs=final_probs,
            generated_ids=generated_ids,
            labels=transformed_labels,
            label_freqs=label_freqs if self.freqs is not None else None,
            sorted_freqs=self.sorted_freqs,
            first_sid_label_locations=first_sid_label_locations,
            max_seq_len=masked_input.shape[1]
        )

    def dense_next_token_prediction(
        self,
        transformed_masked_input: torch.Tensor,
        transformed_labels: torch.Tensor,
        label_locations: torch.Tensor,
    ) -> torch.Tensor:
        bs, seq_len = transformed_masked_input.shape
        num_items = self.codebooks.shape[0]
        self.transformed_codebooks, _ = self.transform_input_and_labels(
            self.codebooks.to(self.device), None, None
        )
        key_embeddings = self.item_sid_embedding_table_encoder.weight[self.transformed_codebooks]
        key_embeddings_norm = key_embeddings.norm(dim=-1).mean(dim=0)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(bs, -1)
        generated_embeddings = torch.zeros(bs, self.num_hierarchies, self.embedding_dim, device=self.device)

        for i in range(self.num_hierarchies):
            # import ipdb; ipdb.set_trace()
            cur_pred_locations = label_locations[i::self.num_hierarchies, :]

            padding_mask = None
            if not self.attend_to_padding:
                new_padding_value = self.num_embeddings_per_hierarchy * self.num_hierarchies
                padding_mask = (transformed_masked_input == new_padding_value)
            
            if i == 0:
                inputs_emb = self.item_sid_embedding_table_encoder.weight[transformed_masked_input]
            output = self.model(inputs_emb, src_key_padding_mask=padding_mask)

            extracted = output[cur_pred_locations[:, 0], cur_pred_locations[:, 1]]
            generated_embeddings[:, i, :] = extracted
            inputs_emb[cur_pred_locations[:, 0], cur_pred_locations[:, 1]] = torch.nn.functional.normalize(extracted, dim=-1) * key_embeddings_norm[i]
        
        return generated_embeddings


    def eval_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        loss_to_aggregate: Optional[torch.Tensor] = None,
    ):
        """
        Evaluation step for the discrete diffusion model.
        """
        
        model_input, label_data = batch
        
        masked_input = model_input.transformed_sequences['sequence_data']
        labels = label_data.labels['sequence_data']
        label_locations = label_data.label_location['sequence_data']

        # Changes label locations to be four values
        modified_masked_input, modified_label_locations = self.get_modified_eval_inputs(
            masked_input, label_locations
        )
        # Create modified data objects
        model_input.transformed_sequences['sequence_data'] = modified_masked_input
        label_data.label_location['sequence_data'] = modified_label_locations
        # Transform both input and labels with hierarchical offsets
        transformed_masked_input, transformed_labels = self.transform_input_and_labels(
            modified_masked_input, labels, modified_label_locations
        )

        if self.diffusion_config['inference_type'] == "beam-search-generation" or self.diffusion_config['inference_type'] == "gen-then-dense":
            generated_ids, final_probs = self.beam_search_generation(
                model_input, label_data
            )
            if self.projection:
                generated_ids = self.project_generated_ids(generated_ids)
            if self.deduplication:
                generated_ids, transformed_labels, final_probs = self.convert_deduplicate(
                    generated_ids, transformed_labels, final_probs, model_input, label_data
                )
            # import ipdb; ipdb.set_trace()
        
        if self.diffusion_config['inference_type'] == "constrained-beam-search-generation":
            # Initialize constrained beam search
            self.constrained_beam_search = ConstrainedBeamSearch(self)
            generated_ids, final_probs = self.constrained_beam_search.beam_search_generation(
                model_input, label_data
            )
        
        if self.diffusion_config['inference_type'] == "dense-retrieval" or self.diffusion_config['inference_type'] == "gen-then-dense":
            bs = transformed_masked_input.shape[0]
            num_items = self.codebooks.shape[0]
            generated_embeddings = self.forward(transformed_masked_input)
            masked_locations = (transformed_masked_input == self.masking_token_id)
            # import ipdb; ipdb.set_trace()
            combined_embeddings = torch.nn.functional.normalize(generated_embeddings[masked_locations], dim=-1)
            combined_embeddings = combined_embeddings.reshape(bs, -1)

            self.transformed_codebooks, _ = self.transform_input_and_labels(
                self.codebooks.to(self.device), None, None
            )
            key_embeddings = self.item_sid_embedding_table_encoder.weight[self.transformed_codebooks]
            key_embeddings = torch.nn.functional.normalize(key_embeddings, dim=-1)
            key_embeddings = key_embeddings.reshape(num_items, -1)

            # convert label_data to match items from transformed_codebooks
            labels_reshaped = labels.view(bs, self.num_hierarchies)
            matches = (labels_reshaped.unsqueeze(1) == self.codebooks.to(self.device).unsqueeze(0))
            item_ids = torch.where(matches.all(dim=2))[1]

        if self.diffusion_config['inference_type'] == "dense-retrieval-with-head":
            bs = transformed_masked_input.shape[0]
            num_items = self.codebooks.shape[0]
            generated_embeddings = self.forward(transformed_masked_input)
            masked_locations = (transformed_masked_input == self.masking_token_id)
            combined_embeddings = generated_embeddings[masked_locations].reshape(bs, -1)

            self.transformed_codebooks, _ = self.transform_input_and_labels(
                self.codebooks.to(self.device), None, None
            )
            labels_reshaped = labels.view(bs, self.num_hierarchies)
            matches = (labels_reshaped.unsqueeze(1) == self.codebooks.to(self.device).unsqueeze(0))
            item_ids = torch.where(matches.all(dim=2))[1]

            key_embeddings = self.item_sid_embedding_table_encoder.weight[self.transformed_codebooks].reshape(num_items, -1)
            dense_retrieval_embeddings = self.dense_retrieval_head(key_embeddings)

        if self.diffusion_config['inference_type'] == "dense-next-token-prediction":
            bs = transformed_masked_input.shape[0]
            num_items = self.codebooks.shape[0]
            self.transformed_codebooks, _ = self.transform_input_and_labels(
                self.codebooks.to(self.device), None, None
            )
            key_embeddings = self.item_sid_embedding_table_encoder.weight[self.transformed_codebooks].reshape(num_items, -1)
            combined_embeddings = self.dense_next_token_prediction(
                transformed_masked_input, transformed_labels, label_locations
            )

            combined_embeddings = combined_embeddings.reshape(bs, -1)
            # convert label_data to match items from transformed_codebooks
            labels_reshaped = labels.view(bs, self.num_hierarchies)
            matches = (labels_reshaped.unsqueeze(1) == self.codebooks.to(self.device).unsqueeze(0))
            item_ids = torch.where(matches.all(dim=2))[1]
            # import ipdb; ipdb.set_trace()

        ## Uncomment this to evaluate the generated ids for small semantic ids
        # transformed_labels = transformed_labels.reshape(-1, self.num_hierarchies)
        # transformed_labels[:, self.eval_hierarchy_cutoff:] = generated_ids[:, 0, self.eval_hierarchy_cutoff:]
        # transformed_labels = transformed_labels.reshape(-1)
        # self.evaluator(
        #     marginal_probs=final_probs,
        #     generated_ids=generated_ids,
        #     labels=transformed_labels,
        #     # label_freqs=label_freqs,
        #     # sorted_freqs=self.sorted_freqs,
        #     # first_sid_label_locations=first_sid_label_locations,
        #     # max_seq_len=masked_input.shape[1]
        # )
        # return

        if self.diffusion_config['inference_type'] == "beam-search-generation" or self.diffusion_config['inference_type'] == "constrained-beam-search-generation":
            self.generative_retrieval_eval(
                generated_ids, masked_input, labels, label_locations, 
                modified_masked_input, modified_label_locations, 
                transformed_labels, final_probs, model_input, label_data
                )
        elif self.diffusion_config['inference_type'] == "dense-retrieval" or self.diffusion_config['inference_type'] == "dense-retrieval-with-head" or self.diffusion_config['inference_type'] == "dense-next-token-prediction":
            # pdb.set_trace()
            self.evaluator(
                query_embeddings=combined_embeddings,
                key_embeddings=key_embeddings,
                labels=item_ids,
                codebook=self.codebooks,
            )
        elif self.diffusion_config['inference_type'] == "gen-then-dense":
            matches = (generated_ids.unsqueeze(2) == self.transformed_codebooks.to(self.device).unsqueeze(0).unsqueeze(0))
            matched_items = matches.all(dim=-1)
            combined_matched_items = matched_items.sum(dim=1).bool()
            self.evaluator(
                query_embeddings=combined_embeddings,
                key_embeddings=key_embeddings,
                labels=item_ids,
                generated_id_mask=combined_matched_items,
                codebook=self.codebooks,
            )


    def transform_input_and_labels(
        self, 
        masked_input: torch.Tensor, 
        labels: torch.Tensor, 
        label_locations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform both masked input and labels by applying hierarchical offsets.
        
        Args:
            masked_input: Input tensor with masked values
            labels: Target labels tensor
            label_locations: Tensor with shape [N, 2] where [:, 0] are batch indices 
                           and [:, 1] are sequence positions
                           
        Returns:
            Tuple of (transformed_masked_input, transformed_labels)
        """
        # Process masked input to replace padding tokens
        if labels is not None:
            assert masked_input.max() < self.num_embeddings_per_hierarchy, "masked_input contains invalid token IDs"
        masked_input = self.process_masked_input(masked_input)

        bs, seq_len = masked_input.shape
        offsets = torch.arange(seq_len, device=self.device) % self.num_hierarchies * self.num_embeddings_per_hierarchy
        
        # Update mask to use the new padding value
        new_padding_value = self.num_embeddings_per_hierarchy * self.num_hierarchies
        mask = (masked_input != self.masking_token_id) & (masked_input != new_padding_value)
        addition = (mask * offsets).to(self.device)

        transformed_masked_input = masked_input + addition

        if labels is None or label_locations is None:
            # If labels or label_locations are not provided, return transformed masked input only
            return transformed_masked_input, None
        
        # Apply the same offset logic to labels based on label_locations
        transformed_labels = labels.clone()
        if len(label_locations) > 0:
            batch_indices = label_locations[:, 0]  
            seq_positions = label_locations[:, 1] 
            
            label_offsets = seq_positions % self.num_hierarchies * self.num_embeddings_per_hierarchy
            transformed_labels += label_offsets.to(self.device)
        
        return transformed_masked_input, transformed_labels


    def get_expanded_candidates(
        self, cur_beam_candidate, cur_beam_values, num_candidates, 
        num_mask_per_row, tokens_to_unmask, labels, label_locations
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, cur_beam_size, seq_len = cur_beam_candidate.shape
        num_combinations = num_candidates ** tokens_to_unmask
        expanded_candidates = torch.zeros(
            (bs, cur_beam_size * num_combinations, seq_len), 
            dtype=torch.long, device=self.device
        )
        expanded_values = torch.zeros(
            (bs, cur_beam_size * num_combinations), 
            dtype=torch.float32, device=self.device
        )
        if tokens_to_unmask > 1:
            assert self.diffusion_config['unmasking_type'] == 'left-to-right', "unmasking_type must be left-to-right when tokens_to_unmask > 1"

        for i in range(cur_beam_size):
            cur_seq = cur_beam_candidate[:, i, :]
            encoder_output = self.forward(cur_seq)
            probabilities = self.encoder_output_to_probabilities(encoder_output)

            if self.diffusion_config['maskout_masking_token_prob_decoding']:
                # Set probabilities of masking token to zero
                probabilities[:, :, self.masking_token_id] = 0.0

            cur_mask = (cur_seq == self.masking_token_id)
            x0 = torch.zeros_like(cur_seq[cur_mask], device=self.device, dtype=torch.long) + self.masking_token_id

            # Select exactly one True position from each row in cur_mask
            # cur_mask shape: (bs, seq_len), each row has self.num_hierarchies True values
            transfer_positions = torch.zeros_like(cur_mask, dtype=torch.bool, device=self.device)
            batch_indices, seq_indices = torch.where(cur_mask)
            
            # Generate random offsets for each batch (vectorized)
            if self.diffusion_config['unmasking_type'] == 'random':
                random_offsets = torch.randint(0, num_mask_per_row, (bs, tokens_to_unmask), device=self.device)
            elif (
                self.diffusion_config['unmasking_type'] == 'top-prob' or 
                self.diffusion_config['unmasking_type'] == 'top-prob-dup-last'
                ):
                # Select the top probability position from each row
                top_values = probabilities[cur_mask].max(dim=-1).values.reshape(bs, -1)
                if self.diffusion_config['unmasking_type'] == 'top-prob-dup-last':
                    top_values[:, -1] = 0.0

                # Vectorized sampling from top_values using temperature
                temperature = self.diffusion_config['unmasking_temperature']
                logits = top_values / temperature  # (bs, num_mask_per_row)
                probs = torch.softmax(logits, dim=1)
                random_offsets = torch.multinomial(probs, tokens_to_unmask)  # (bs, tokens_to_unmask)
            elif self.diffusion_config['unmasking_type'] == 'left-to-right':
                # Generate left-to-right offsets - select the first tokens_to_unmask masked positions
                # This should be relative to the flattened masked positions array
                random_offsets = torch.arange(tokens_to_unmask, device=self.device, dtype=torch.long).unsqueeze(0).expand(bs, tokens_to_unmask).clone()
            

            # pdb.set_trace()
            # Calculate indices in the flattened True positions array
            selected_indices = torch.arange(bs, device=self.device)[:, None] * num_mask_per_row + random_offsets
            selected_indices = selected_indices.flatten()

            # Get the selected positions
            selected_batch_indices = batch_indices[selected_indices]
            selected_seq_indices = seq_indices[selected_indices]
            
            # Set the selected positions to True
            transfer_positions[selected_batch_indices, selected_seq_indices] = True
            
            # Create transfer_index_t_s for the masked positions only
            transfer_index_t_s = transfer_positions[cur_mask]
            probs_at_transfer = probabilities[cur_mask][transfer_index_t_s]
            probs_at_transfer = probs_at_transfer.reshape(bs, tokens_to_unmask, -1)

            # Convert probs_at_transfer of shape (total_transfer_positions, num_candidates) to (bs, tokens_to_unmask, num_candidates)
            # probs_at_transfer = probs_at_transfer.reshape(bs, tokens_to_unmask, total_size)
            # create an expanded_candidates of shape (bs, num_candidates ** tokens_to_unmask) by just iterating over the num_candidates ** tokens_to_unmask combinations
            
            assert num_candidates <= probs_at_transfer.shape[-1], f"num_candidates {num_candidates} exceeds vocab size {probs_at_transfer.shape[-1]}"
            
            # # (total_transfer_positions, num_candidates)
            topk_values, topk_indices = torch.topk(probs_at_transfer, num_candidates, dim=-1) #(bs, tokens_to_unmask, num_candidates)
            # transfer_row_index = torch.where(cur_mask)[0][transfer_index_t_s]
            # Create all possible combinations of candidate indices for the tokens to unmask
            # expanded_candidates: (bs, num_combinations, seq_len)
            # expanded_values: (bs, num_combinations)
            

            # Generate all possible combinations of candidate indices for the tokens to unmask
            candidate_combinations = list(product(range(num_candidates), repeat=tokens_to_unmask))  # (num_combinations, tokens_to_unmask)

            # transfer_index_t_s: (bs, tokens_to_unmask)
            # topk_indices: (bs, tokens_to_unmask, num_candidates)
            # topk_values: (bs, tokens_to_unmask, num_candidates)
            for combo_idx, combo in enumerate(candidate_combinations):
                # For each token to unmask, get the candidate index and value for this combination
                indices = []
                values = []
                for t in range(tokens_to_unmask):
                    indices.append(topk_indices[:, t, combo[t]])  # (bs,)
                    values.append(topk_values[:, t, combo[t]])    # (bs,)
                
                indices = torch.stack(indices, dim=1)    # (bs, tokens_to_unmask)
                values = torch.stack(values, dim=1)    # (bs, tokens_to_unmask)


                x0_candidate = x0.clone()
                x0_candidate[transfer_index_t_s] = indices.flatten().to(torch.long).to(self.device)
                
                candidate_prob = cur_beam_values[:, i].clone()
                candidate_prob = candidate_prob * values.prod(dim=1)

                new_seq = cur_seq.clone()
                new_seq[cur_mask] = x0_candidate
                
                expanded_candidates[:, i * num_combinations + combo_idx, :] = new_seq
                expanded_values[:, i * num_combinations + combo_idx] = candidate_prob
                # if tokens_to_unmask > 1:
                #     pdb.set_trace()

                # # Set the selected candidate indices at the transfer positions
                # # transfer_index_t_s: (bs, tokens_to_unmask)
                # batch_idx = torch.arange(bs, device=x0.device).unsqueeze(1).expand(bs, tokens_to_unmask)
                # expanded_candidates[selected_batch_indices, i * num_combinations + combo_idx, selected_seq_indices] = indices

                # # Compute the product of the selected probabilities for each combination
                # expanded_values[:, i * num_combinations + combo_idx] = values.prod(dim=1)

            # for j in range(num_candidates):
            #     x0_candidate = x0.clone()                           #(total_masked_positions,)
            #     candidate_prob = cur_beam_values[:, i].clone()      #(bs,)

            #     x0_candidate[transfer_index_t_s] = topk_indices[:, j]
            #     prob_updates = torch.ones_like(candidate_prob)
            #     # prob_updates.scatter_reduce_(0, transfer_row_index, topk_values[:, j], reduce='prod', include_self=False)
            #     prob_updates.scatter_reduce_(0, selected_batch_indices, topk_values[:, j], reduce='prod', include_self=False)

            #     new_seq = cur_seq.clone()
            #     new_seq[cur_mask] = x0_candidate
                
            #     expanded_candidates[:, i * num_candidates + j, :] = new_seq
            #     expanded_values[:, i * num_candidates + j] = candidate_prob * prob_updates
        
        
        return expanded_candidates, expanded_values

    def beam_search_generation(
        self,
        model_input: torch.Tensor,
        label_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs beam search to generate candidate sequences from masked positions.
        
        Args:
            probabilities: Model output probabilities with shape (bs, seq_len, vocab_size)
            modified_masked_input: Input tensor with masked tokens
            num_candidates: Number of top candidates to keep during beam search
            
        Returns:
            Tuple containing:
                - generated_ids: Generated token sequences (bs, num_candidates, num_hierarchies)
                - beam_values: Final probability scores for each candidate (bs, num_candidates)
        """
        num_candidates = self.diffusion_config['num_candidates']
        steps = self.diffusion_config['num_steps']
        
        timesteps = torch.linspace(1, 1e-5, steps + 1, device=self.device)
        bs, seq_len = model_input.transformed_sequences['sequence_data'].shape
        generated_ids = torch.zeros(
            (bs, num_candidates, self.num_hierarchies), 
            dtype=torch.long, device=self.device)
        probs = torch.ones((bs, num_candidates), device=self.device)

        label_locations = label_data.label_location['sequence_data']
        batch_indices, seq_indices = label_locations[:, 0], label_locations[:, 1]
        transformed_masked_input, transformed_labels = self.transform_input_and_labels( 
            model_input.transformed_sequences['sequence_data'], label_data.labels['sequence_data'], label_locations 
            )

        cur_beam_candidate = torch.zeros((bs, 1, seq_len), dtype=torch.long, device=self.device)
        cur_beam_values = torch.ones((bs, 1), dtype=torch.float32, device=self.device)
        cur_beam_candidate[:, 0, :] = transformed_masked_input

        num_mask_per_row = self.num_hierarchies
        if 'unmasking_nums' in self.diffusion_config and self.diffusion_config['unmasking_nums'] is not None:
            steps = len(self.diffusion_config['unmasking_nums'])

        for j in range(steps):
            if 'unmasking_nums' in self.diffusion_config and self.diffusion_config['unmasking_nums'] is not None:
                tokens_to_unmask = self.diffusion_config['unmasking_nums'][j]
            else:
                tokens_to_unmask = self.num_hierarchies // steps
            
            expanded_candidates, expanded_values = self.get_expanded_candidates(
                cur_beam_candidate, cur_beam_values, num_candidates, 
                num_mask_per_row, tokens_to_unmask, transformed_labels, label_locations
            )
            num_mask_per_row -= tokens_to_unmask
            
            # Select top num_candidates based on values
            topk_values, topk_indices = torch.topk(expanded_values, num_candidates, dim=-1)
            new_beam_candidate = torch.zeros((bs, num_candidates, seq_len), dtype=torch.long, device=self.device)
            new_beam_values = torch.zeros((bs, num_candidates), dtype=torch.float32, device=self.device)

            for i in range(num_candidates):
                new_beam_candidate[:, i, :] = expanded_candidates[torch.arange(bs), topk_indices[:, i], :]
                new_beam_values[:, i] = expanded_values[torch.arange(bs), topk_indices[:, i]]

            cur_beam_candidate = new_beam_candidate
            cur_beam_values = new_beam_values
            # if j == self.eval_hierarchy_cutoff - 1:
            #     break
            
            # pdb.set_trace()

        for i in range(num_candidates):
            cur_seq = cur_beam_candidate[:, i, :]
            generated_ids_flatten = cur_seq[batch_indices, seq_indices]
            generated_ids[:, i, :] = generated_ids_flatten.reshape(-1, self.num_hierarchies)
        
        return generated_ids, new_beam_values


    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set."""
        self.eval_step(batch, self.val_loss)
        self.log("val_loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Perform a single test step on a batch of data from the test set."""
        self.eval_step(batch, self.test_loss)
        self.log("test_loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)


    def mask_input_and_labels(self, model_input, label_data):
        """
        Mask random fractions of positions in each row of input sequences.
        
        Args:
            model_input: Model input data containing sequences
            label_data: Label data containing labels and locations
            
        Returns:
            model_input, label_data: Modified input and label data with masking applied
        """
        input_seq = model_input.transformed_sequences['sequence_data']
        labels_seq = label_data.labels['sequence_data']
        label_locations_seq = label_data.label_location['sequence_data']
        
        batch_size, seq_len = input_seq.shape
        device = input_seq.device
        unpadded_positions = (input_seq != self.padding_token_id)
        
        # Generate random fractions for each row (uniform between 0 and 1)
        cur_noise_schedule = self.diffusion_config['noise_schedule']
        if self.trainer.global_step % self.dense_retrieval_update_frequency == 0 and self.use_dense_retrieval_head:
            cur_noise_schedule = 'last-token-ar'
        if cur_noise_schedule == 'uniform':
            fractions = torch.rand(batch_size, device=device) * self.diffusion_config['max_mask_fraction']
        elif cur_noise_schedule == 'edm':
            snr = (1.2 * torch.randn(batch_size, device=device) - 1.2).exp()
            fractions = snr / (snr + 1)
        elif cur_noise_schedule == 'last-token-ar':
            idx = torch.arange(input_seq.size(1), device=input_seq.device).unsqueeze(0).expand_as(input_seq)
            idx_padded = idx * unpadded_positions + (~unpadded_positions) * -1
            sorted_idx = torch.argsort(idx_padded, dim=1, descending=True)
            last_k_idx = sorted_idx[:, :self.num_hierarchies]

            unpadded_mask_positions = torch.zeros_like(input_seq, dtype=torch.bool)
            row_idx = torch.arange(input_seq.size(0), device=input_seq.device).unsqueeze(1).expand_as(last_k_idx)
            unpadded_mask_positions[row_idx, last_k_idx] = True
            fractions = torch.ones(input_seq.size(0), dtype=torch.float32, device=input_seq.device)
            # fractions = self.num_hierarchies / unpadded_positions.sum(dim=-1)
        

        if cur_noise_schedule == 'edm' or cur_noise_schedule == 'uniform':
            random_vals = torch.rand(batch_size, seq_len, device=device)
        
            # Create mask tensor
            mask_positions = random_vals < fractions.unsqueeze(1)
            unpadded_mask_positions = mask_positions & unpadded_positions
        
        # Create masked input
        masked_input = input_seq.clone()
        masked_input[unpadded_mask_positions] = self.masking_token_id
        
        labels = input_seq[unpadded_mask_positions]
        batch_indices, col_indices = torch.where(unpadded_mask_positions)
        label_locations = torch.vstack((batch_indices, col_indices)).t()
        
        # Update the data structures
        model_input.transformed_sequences['sequence_data'] = masked_input
        label_data.labels['sequence_data'] = labels
        label_data.label_location['sequence_data'] = label_locations
        
        # import pdb; pdb.set_trace()
        return model_input, label_data, fractions

    def on_load_checkpoint(self, checkpoint):
        """Override checkpoint loading to reset learning rate and scheduler state."""
        # Call parent method first

        checkpoint['lr_schedulers'][0]['min_ratio'] = self.scheduler.keywords['min_ratio']
        checkpoint['lr_schedulers'][0]['base_lrs'][0] = self.optimizer.keywords['lr']
        checkpoint['lr_schedulers'][0]['scheduler_steps'] = self.scheduler.keywords['scheduler_steps']
        
        # Handle missing dense_retrieval_head parameters by initializing them randomly
        if self.use_dense_retrieval_head and self.dense_retrieval_head is not None:
            state_dict = checkpoint['state_dict']
            model_state_dict = self.state_dict()
            
            # Check for missing dense_retrieval_head parameters
            missing_keys = []
            for key in model_state_dict.keys():
                if key.startswith('dense_retrieval_head.') and key not in state_dict:
                    missing_keys.append(key)
            
            # import ipdb; ipdb.set_trace()
            if missing_keys:
                print(f"Missing dense_retrieval_head parameters: {missing_keys}")
                print("Initializing missing parameters randomly...")
                
                # Initialize missing parameters randomly
                for key in missing_keys:
                    param_shape = model_state_dict[key].shape
                    param_dtype = model_state_dict[key].dtype
                    param_device = model_state_dict[key].device
                    
                    # Use Xavier uniform initialization for linear layers
                    if 'weight' in key:
                        # Xavier uniform initialization
                        bound = (6.0 / (param_shape[0] + param_shape[1])) ** 0.5
                        state_dict[key] = torch.empty(param_shape, dtype=param_dtype, device=param_device).uniform_(-bound, bound)
                    elif 'bias' in key:
                        # Initialize bias to zero
                        state_dict[key] = torch.zeros(param_shape, dtype=param_dtype, device=param_device)
                    else:
                        # Default random initialization
                        state_dict[key] = torch.randn(param_shape, dtype=param_dtype, device=param_device) * 0.1
                
                # Update the checkpoint with the new parameters
                checkpoint['state_dict'] = state_dict
                print("Successfully initialized missing dense_retrieval_head parameters")
                
                # Handle optimizer state for new parameters
                # Option 1: Clear optimizer state entirely (simpler and more reliable)
                if 'optimizer_states' in checkpoint:
                    print("Clearing optimizer state due to new parameters...")
                    checkpoint['optimizer_states'] = []
                    print("Successfully cleared optimizer state - optimizer will be reinitialized")
                
                # Option 2: Try to patch optimizer state (more complex, commented out for now)
                # if 'optimizer_states' in checkpoint and len(checkpoint['optimizer_states']) > 0:
                #     print("Handling optimizer state for new parameters...")
                #     optimizer_state = checkpoint['optimizer_states'][0]
                #     
                #     # Get the current model parameters
                #     current_params = list(self.parameters())
                #     print(f"Current model has {len(current_params)} parameters")
                #     
                #     # Find which parameters are new (dense_retrieval_head parameters)
                #     new_param_indices = []
                #     for i, (name, param) in enumerate(self.named_parameters()):
                #         if name.startswith('dense_retrieval_head.'):
                #             new_param_indices.append(i)
                #             print(f"Found new parameter: {name} at index {i}")
                #     
                #     print(f"New parameter indices: {new_param_indices}")
                #     
                #     # For each new parameter, we need to add empty state to optimizer
                #     if 'state' in optimizer_state:
                #         for param_idx in new_param_indices:
                #             if param_idx not in optimizer_state['state']:
                #                 optimizer_state['state'][param_idx] = {}
                #                 print(f"Added empty optimizer state for parameter {param_idx}")
                #     
                #     # Update param_groups to include new parameters
                #     if 'param_groups' in optimizer_state and len(optimizer_state['param_groups']) > 0:
                #         param_group = optimizer_state['param_groups'][0]
                #         if 'params' in param_group:
                #             print(f"Original param_groups params: {param_group['params']}")
                #             # Add new parameter indices to the param_groups
                #             existing_params = set(param_group['params'])
                #             for param_idx in new_param_indices:
                #                 if param_idx not in existing_params:
                #                     param_group['params'].append(param_idx)
                #                     print(f"Added parameter {param_idx} to optimizer param_groups")
                #             print(f"Updated param_groups params: {param_group['params']}")
                #     
                #     checkpoint['optimizer_states'][0] = optimizer_state
                #     print("Successfully updated optimizer state for new parameters")
        
        super().on_load_checkpoint(checkpoint)

