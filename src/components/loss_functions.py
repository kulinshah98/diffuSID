from typing import Optional

import torch
import torch.nn.functional as F


class InBatchContrastiveLoss(torch.nn.Module):
    """Contrastive loss with in-batch negative samples for item prediction task.

    We implement the InfoNCE loss with same-item negative-pair masking for in-batch
    prediction tasks.

    For reference on the InfoNCE loss, see for example Equation 6 in
    https://dl.acm.org/doi/pdf/10.5555/3157096.3157304
    """

    def __init__(
        self,
        contrastive_tau: float = 0.1,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Initialize the InBatchContrastiveLoss.

        Parameters
        ----------
        contrastive_tau: float
            Temperature parameter for the contrastive loss.
        normalize: bool
            Whether to normalize the embeddings before computing the logits via dot product.
        """
        super().__init__()
        self.tau = contrastive_tau
        self.normalize = normalize

    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        label_locations: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the contrastive loss with in-batch negative samples.

        Parameters
        ----------
        query_embeddings: torch.Tensor (batch_size x sequence length x embedding_dim)
            The embeddings of the query items.
        key_embeddings: torch.Tensor (total number of items or number of labels x embedding_dim)
            The embeddings of the key items.
        label_locations: torch.Tensor (number of labels x 2)
            The locations of the labels in the input sequences.
        labels: torch.Tensor (number of labels)
            The labels for the input sequences.

        Returns
        -------
        torch.Tensor
            The contrastive loss.
        """
        labels = labels.to(query_embeddings.device)
        if len(labels) != len(key_embeddings):
            key_embeddings = key_embeddings[labels]
        if label_locations is not None:
            # get representations of masked tokens
            # label_locations[:, 0] refers to the index of sequences
            # label_locations[:, 1] refers to the index of tokens in the sequences
            label_locations = label_locations.to(query_embeddings.device)

            query_embeddings = query_embeddings[
                label_locations[:, 0], label_locations[:, 1]
            ]

        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, dim=-1)
            key_embeddings = F.normalize(key_embeddings, dim=-1)

        # construct a mask to remove negative pairs for which the key and query item are the same
        # this mask will be applied to the logits obtained by multiplying query and key embeddings
        mask = labels.expand(labels.shape[0], -1) != labels.reshape(-1, 1)
        # diagonal elements correspond to positive pairs and should not be masked out
        diagonal_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=mask.device)
        mask = mask | diagonal_mask

        # compute the InfoNCE loss with masking of negative samples with identical items
        logits = torch.mm(query_embeddings, key_embeddings.t())
        numerator = torch.exp(torch.diagonal(logits) / self.tau)
        # Masking happens before the exp, so the masked terms will be 1 in the denominator.
        # We choose this implementation because it less drastically changes the
        # denominator than masking after the exp, which would set the masked terms to 0.
        # We expect that with true negatives, the terms would be closer to 1 than 0.
        denominator = torch.sum(torch.exp(torch.mul(logits, mask) / self.tau), dim=-1)
        loss = -torch.log(numerator / denominator)

        return loss.mean()

class FullDenseRetrievalLoss(torch.nn.Module):
    """
    Dense retrieval loss.
    """
    
    
    def __init__(
        self,
        contrastive_tau: float = 0.1,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.tau = contrastive_tau
        self.normalize = normalize
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        
    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the contrastive loss with negative samples from the full vocabulary.
        """
        # query embeddings shape: (batch_size, embedding_dim)
        # key embeddings shape: (total number of items, embedding_dim)
        logits = torch.mm(query_embeddings, key_embeddings.t()) / self.tau

        loss = self.cross_entropy_loss(logits, labels.long())

        return loss.mean()
        


class FullBatchCrossEntropyLoss(torch.nn.Module):
    """
    Contrastive loss with negative samples being all candidates in the embedding table.
    """

    def __init__(
        self,
        contrastive_tau: float = 0.1,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Initialize the FullBatchContrastiveLoss.

        Parameters
        ----------
        contrastive_tau: float
            Temperature parameter for the contrastive loss.
        normalize: bool
            Whether to normalize the embeddings before computing the logits via dot product.
        """
        super().__init__()
        self.normalize = normalize
        self.tau = contrastive_tau
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(
        self,
        query_embeddings: torch.Tensor,
        key_embeddings: torch.Tensor,
        label_locations: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the contrastive loss with negative samples from the full vocabulary.

        Parameters
        ----------
        query_embeddings: torch.Tensor (batch_size x sequence length x embedding_dim)
            The embeddings of the query items.
        key_embeddings: torch.Tensor (total number of items x embedding_dim)
            The embeddings of all items, i.e the full embedding table.
        label_locations: torch.Tensor (number of labels x 2)
            The locations of the labels in the input sequences.
        labels: torch.Tensor (number of labels)
            The labels for the input sequences.

        Returns
        -------
        torch.Tensor
            The contrastive loss.
        """
        # get representation of masked tokens
        # label_locations[:, 0] refers to the index of sequences
        # label_locations[:, 1] refers to the index of tokens in the sequences
        query_embeddings = query_embeddings[
            label_locations[:, 0], label_locations[:, 1]
        ]

        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, dim=-1)
            key_embeddings = F.normalize(key_embeddings, dim=-1)

        logits = torch.mm(query_embeddings, key_embeddings.t()) / self.tau

        loss = self.cross_entropy_loss(logits, labels.long())

        return (loss * weights).sum()

