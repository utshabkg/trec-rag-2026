"""
Base model classes and interfaces for QDER models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class BaseQDERModel(nn.Module, ABC):
    """
    Abstract base class for QDER models.

    Defines the common interface that all QDER model variants should implement.
    """

    def __init__(self,
                 pretrained: str,
                 use_scores: bool = True,
                 use_entities: bool = True,
                 score_method: str = 'linear'):
        """
        Initialize base QDER model.

        Args:
            pretrained: Name of pretrained text encoder
            use_scores: Whether to use document scores
            use_entities: Whether to use entity embeddings
            score_method: Scoring method ('linear' or 'bilinear')
        """
        super(BaseQDERModel, self).__init__()
        self.pretrained = pretrained
        self.use_scores = use_scores
        self.use_entities = use_entities
        self.score_method = score_method

        # Model configuration
        self.config = {
            'pretrained': pretrained,
            'use_scores': use_scores,
            'use_entities': use_entities,
            'score_method': score_method
        }

    @abstractmethod
    def forward(self,
                query_input_ids: torch.Tensor,
                query_attention_mask: torch.Tensor,
                query_token_type_ids: torch.Tensor,
                query_entity_emb: torch.Tensor,
                doc_input_ids: torch.Tensor,
                doc_attention_mask: torch.Tensor,
                doc_token_type_ids: torch.Tensor,
                doc_entity_emb: torch.Tensor,
                query_entity_mask: Optional[torch.Tensor] = None,
                doc_entity_mask: Optional[torch.Tensor] = None,
                doc_score: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            query_input_ids: Query token IDs [batch_size, seq_len]
            query_attention_mask: Query attention mask [batch_size, seq_len]
            query_token_type_ids: Query token type IDs [batch_size, seq_len]
            query_entity_emb: Query entity embeddings [batch_size, num_entities, emb_dim]
            doc_input_ids: Document token IDs [batch_size, seq_len]
            doc_attention_mask: Document attention mask [batch_size, seq_len]
            doc_token_type_ids: Document token type IDs [batch_size, seq_len]
            doc_entity_emb: Document entity embeddings [batch_size, num_entities, emb_dim]
            query_entity_mask: Query entity attention mask [batch_size, num_entities]
            doc_entity_mask: Document entity attention mask [batch_size, num_entities]
            doc_score: Document retrieval scores [batch_size]

        Returns:
            Dictionary containing 'score' and optionally other outputs
        """
        pass

    @abstractmethod
    def get_text_embeddings(self,
                            input_ids: torch.Tensor,
                            attention_mask: torch.Tensor,
                            token_type_ids: torch.Tensor,
                            encoder_type: str = 'query') -> torch.Tensor:
        """
        Get text embeddings from the specified encoder.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            encoder_type: 'query' or 'doc' encoder

        Returns:
            Text embeddings [batch_size, seq_len, hidden_size]
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model configuration.

        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.__class__.__name__,
            'pretrained': self.pretrained,
            'use_scores': self.use_scores,
            'use_entities': self.use_entities,
            'score_method': self.score_method,
            'num_parameters': self.count_parameters(),
            'trainable_parameters': self.count_trainable_parameters()
        }

    def count_parameters(self) -> int:
        """Count total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_text_encoders(self) -> None:
        """Freeze the text encoder parameters."""
        if hasattr(self, 'query_encoder'):
            for param in self.query_encoder.parameters():
                param.requires_grad = False
        if hasattr(self, 'doc_encoder'):
            for param in self.doc_encoder.parameters():
                param.requires_grad = False

    def unfreeze_text_encoders(self) -> None:
        """Unfreeze the text encoder parameters."""
        if hasattr(self, 'query_encoder'):
            for param in self.query_encoder.parameters():
                param.requires_grad = True
        if hasattr(self, 'doc_encoder'):
            for param in self.doc_encoder.parameters():
                param.requires_grad = True

    def get_attention_weights(self,
                              query_text_emb: torch.Tensor,
                              doc_text_emb: torch.Tensor,
                              query_entity_emb: Optional[torch.Tensor] = None,
                              doc_entity_emb: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute attention weights between query and document representations.

        Args:
            query_text_emb: Query text embeddings
            doc_text_emb: Document text embeddings
            query_entity_emb: Query entity embeddings (optional)
            doc_entity_emb: Document entity embeddings (optional)

        Returns:
            Dictionary with attention weights
        """
        attention_weights = {}

        # Text attention
        text_attention_scores = torch.matmul(query_text_emb, doc_text_emb.transpose(-2, -1))
        text_attention_probs = torch.softmax(text_attention_scores, dim=-1)
        attention_weights['text_attention'] = text_attention_probs

        # Entity attention (if available)
        if (query_entity_emb is not None and doc_entity_emb is not None and
                query_entity_emb.size(1) > 0 and doc_entity_emb.size(1) > 0):
            entity_attention_scores = torch.matmul(query_entity_emb, doc_entity_emb.transpose(-2, -1))
            entity_attention_probs = torch.softmax(entity_attention_scores, dim=-1)
            attention_weights['entity_attention'] = entity_attention_probs

        return attention_weights

    def save_pretrained(self, save_path: str) -> None:
        """
        Save model state and configuration.

        Args:
            save_path: Path to save the model
        """
        import json
        import os

        os.makedirs(save_path, exist_ok=True)

        # Save model state
        torch.save(self.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))

        # Save configuration
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load model from saved state.

        Args:
            model_path: Path to saved model
            **kwargs: Additional arguments for model initialization

        Returns:
            Loaded model instance
        """
        import json
        import os

        # Load configuration
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            config.update(kwargs)  # Override with any provided kwargs
        else:
            config = kwargs

        # Create model instance
        model = cls(**config)

        # Load state dict
        state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location='cpu')
            model.load_state_dict(state_dict)

        return model