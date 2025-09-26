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
from src.utils.file_utils import open_local_or_remote
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



class DenseItemHeadDDModule(BaseModule):
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
        embedding_path: str,
        dense_retrieval: Optional[dict] = None,
        dense_retrieval_loss_function: Optional[object] = None,
        item_embedding_projection: Optional[torch.nn.Module] = None,
        dense_retrieval_evaluator: Optional[object] = None,
        normalization: Optional[bool] = True,
        positional_embedding: Optional[torch.nn.Module] = None,
        attend_to_padding: Optional[bool] = True,
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

        my_evaluator = evaluator
        if diffusion_config['inference_type'] == "dense-retrieval":
            my_evaluator = dense_retrieval_evaluator
        
        
        super().__init__(
            model=model,
            optimizer=optimizer, 
            scheduler=scheduler,
            loss_function=loss_function,
            evaluator=my_evaluator,
            training_loop_function=training_loop_function
        )
        
        self.positional_embedding = positional_embedding
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
        self.update_frequency = update_frequency
        
        self.dense_retrieval = dense_retrieval
        self.dense_retrieval_loss_function = dense_retrieval_loss_function
        self.item_embedding_projection = item_embedding_projection
        self.dense_retrieval_evaluator = dense_retrieval_evaluator
        
        self.num_embeddings_per_hierarchy = vocab_size + 1  # +1 for masking
        self.embedding_dim = embedding_dim

        fs = gcsfs.GCSFileSystem()
        self.freqs = None
        self.sorted_freqs = None
        if data_freqs_path is not None:
            with open(data_freqs_path, "rb") as f:
                data = pickle.load(f)

            self.items = torch.Tensor(data["items"])
            self.freqs = torch.Tensor(data["freqs"])
            self.sorted_freqs = self.freqs.clone()
            self.sorted_freqs = self.sorted_freqs.sort().values
        
        self.item_embeddings = torch.load(open_local_or_remote(embedding_path, "rb"))
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
        input: SequentialModelInputData, 
        give_half_way_embedding: Optional[bool] = False):
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
            
            inputs_emb = encoded_input + pos_embeddings
        
        padding_mask = None
        if not self.attend_to_padding:
            new_padding_value = self.num_embeddings_per_hierarchy * self.num_hierarchies
            padding_mask = (input == new_padding_value)
        
        if give_half_way_embedding:
            output, half_way_embedding = self.model(inputs_emb, src_key_padding_mask=padding_mask, give_half_way_embedding=True)
        else:
            output = self.model(inputs_emb, src_key_padding_mask=padding_mask)

        # pdb.set_trace()
        if self.normalization:
            output = output / math.sqrt(self.embedding_dim)

        if give_half_way_embedding:
            return output, half_way_embedding
        else:
            return output
        


    def get_dense_retrieval_loss(
        self,
        encoder_output: torch.Tensor,
        transformed_masked_input: torch.Tensor,
        transformed_labels: torch.Tensor,
        label_locations: torch.Tensor,
        halfway_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bs = encoder_output.shape[0]
        self.transformed_codebooks, _ = self.transform_input_and_labels(
            self.codebooks.to(self.device), None, None        
        )
        transformed_masked_input_clone = transformed_masked_input.clone()
        transformed_masked_input_clone[ label_locations[:, 0], label_locations[:, 1] ] = transformed_labels
        nonpad_items = torch.where( transformed_masked_input_clone != self.num_embeddings_per_hierarchy * self.num_hierarchies )
        all_items = transformed_masked_input_clone[ nonpad_items ].reshape(-1, self.num_hierarchies)

        matches = (all_items.unsqueeze(1) == self.transformed_codebooks.to(self.device).unsqueeze(0))
        item_ids = torch.where(matches.all(dim=2))[1]

        if self.dense_retrieval['apply_halfway_embedding']:
            concated_embeddings = halfway_output[nonpad_items[0], nonpad_items[1]]
        
        if self.dense_retrieval['aggregation'] == "concat":
            concated_embeddings = concated_embeddings.reshape(len(item_ids), -1)
        
        key_embeddings = self.item_embedding_projection(self.item_embeddings.to(self.device))
        return self.dense_retrieval_loss_function(
            query_embeddings=concated_embeddings,
            key_embeddings=key_embeddings,
            labels=item_ids,
        )


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

        if self.dense_retrieval['apply_halfway_embedding']:
            encoder_output, halfway_output = self.forward(transformed_masked_input, give_half_way_embedding=True)
        else:
            encoder_output = self.forward(transformed_masked_input, give_half_way_embedding=False)

        if self.diffusion_config['loss_reweighting'] == 'equal':
            weights = torch.ones_like(transformed_labels, dtype=torch.float32)
        elif self.diffusion_config['loss_reweighting'] == 'normalized':
            weights = 1.0 / fractions[ torch.where(masked_locations)[0] ]

        loss = self.loss_function(
            query_embeddings=encoder_output, 
            key_embeddings=self.item_sid_embedding_table_encoder.weight,
            label_locations=label_locations,
            labels=transformed_labels,
            weights=weights,
            )
        
        # Always use item_embedding_projection to ensure all parameters receive gradients
        # This prevents DDP errors from unused parameters    
        avg_loss = loss / torch.sum(weights)
        
        if self.dense_retrieval_loss_function is not None and self.trainer.global_step % self.dense_retrieval['update_frequency'] == 0:
            avg_loss += self.dense_retrieval['loss_weight'] * self.get_dense_retrieval_loss(
                encoder_output=encoder_output,
                transformed_masked_input=transformed_masked_input,
                transformed_labels=transformed_labels,
                label_locations=label_locations,
                halfway_output=halfway_output if self.dense_retrieval['apply_halfway_embedding'] else None
                )
        else:
            concated_embeddings = torch.zeros(masked_input.shape[0], self.embedding_dim * self.num_hierarchies, device=self.device)
            
            if self.dense_retrieval['aggregation'] == "mean":
                concated_embeddings = concated_embeddings.reshape(masked_input.shape[0], -1, self.num_hierarchies)
                concated_embeddings = concated_embeddings.mean(dim=-1)
            elif self.dense_retrieval['aggregation'] == "concat":
                concated_embeddings = concated_embeddings.reshape(masked_input.shape[0], -1)
            
            dummy_projection = self.item_embedding_projection(self.item_embeddings.to(self.device))
            # Add a small zero contribution to ensure gradient flow
            avg_loss = avg_loss + 0.0 * dummy_projection.sum()
        
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

        if self.diffusion_config['inference_type'] == "beam-search-generation":
            generated_ids, final_probs = self.beam_search_generation(
                model_input, label_data
            )
            if self.projection:
                generated_ids = self.project_generated_ids(generated_ids)
            if self.deduplication:
                generated_ids, transformed_labels, final_probs = self.convert_deduplicate(
                    generated_ids, transformed_labels, final_probs, model_input, label_data
                )
        
        if self.diffusion_config['inference_type'] == "dense-retrieval":
            bs = transformed_masked_input.shape[0]
            if self.dense_retrieval['apply_halfway_embedding']:
                encoder_output, halfway_output = self.forward(transformed_masked_input, give_half_way_embedding=True)
            else:
                encoder_output = self.forward(transformed_masked_input, give_half_way_embedding=False)
            
            if self.dense_retrieval['apply_halfway_embedding']:
                generated_embeddings = halfway_output
            else:
                generated_embeddings = encoder_output
            
            input_embeddings = generated_embeddings[label_locations[:, 0], label_locations[:, 1]].reshape(bs, -1)
            
            if self.dense_retrieval['aggregation'] == "mean":
                input_embeddings = input_embeddings.reshape(bs, -1, self.num_hierarchies)
                input_embeddings = input_embeddings.mean(dim=-1)
            elif self.dense_retrieval['aggregation'] == "concat":
                input_embeddings = input_embeddings.reshape(bs, -1)
            
            query_embeddings = input_embeddings
            key_embeddings = self.item_embedding_projection(self.item_embeddings.to(self.device))

            self.transformed_codebooks, _ = self.transform_input_and_labels(
                self.codebooks.to(self.device), None, None
            )

            transformed_labels_reshaped = transformed_labels.view(bs, self.num_hierarchies)
            matches = (transformed_labels_reshaped.unsqueeze(1) == self.transformed_codebooks.to(self.device).unsqueeze(0))
            item_ids = torch.where(matches.all(dim=2))[1]


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
        if self.diffusion_config['inference_type'] == "dense-retrieval":
            self.evaluator(
                query_embeddings=query_embeddings.to(self.device), 
                key_embeddings=key_embeddings.to(self.device), 
                labels=item_ids.to(self.device), 
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
            encoder_output = self.forward(cur_seq, give_half_way_embedding=False)
            # import pdb; pdb.set_trace()
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
        
        if self.trainer.global_step % self.dense_retrieval['update_frequency'] == 0 and self.dense_retrieval['noise'] != 'apply-all-items':
            cur_noise_schedule = self.dense_retrieval['noise']
        
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
        elif cur_noise_schedule == 'mask-random-item':
            cur_num_items = unpadded_positions.sum(dim=-1) // self.num_hierarchies
            mask_item_index = (torch.rand_like(cur_num_items.float()) * cur_num_items).long()
            mask_positions = (mask_item_index[:, None] * self.num_hierarchies) + torch.arange(self.num_hierarchies, device=device)[None, :]
            unpadded_mask_positions = torch.zeros_like(input_seq, dtype=torch.bool)
            unpadded_mask_positions[ torch.arange(batch_size, device=device)[:, None], mask_positions ] = True
            fractions = torch.ones(batch_size, dtype=torch.float32, device=device)

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
        
        super().on_load_checkpoint(checkpoint)

