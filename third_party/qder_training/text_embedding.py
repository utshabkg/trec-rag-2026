"""
Text embedding components for QDER models.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, DistilBertModel, T5EncoderModel
from typing import Optional


class TextEmbedding(nn.Module):
    """
    Text embedding component that wraps various pretrained encoders.

    Supports BERT, DistilBERT, RoBERTa, DeBERTa, ERNIE, ELECTRA, ConvBERT, and T5.
    """

    def __init__(self, pretrained: str) -> None:
        """
        Initialize text embedding component.

        Args:
            pretrained: Name/path of pretrained model
        """
        super(TextEmbedding, self).__init__()
        self.pretrained = pretrained
        self.config = AutoConfig.from_pretrained(self.pretrained)

        # Load appropriate model based on architecture
        if pretrained == 't5-base':
            self.encoder = T5EncoderModel.from_pretrained(self.pretrained, config=self.config)
        else:
            self.encoder = AutoModel.from_pretrained(self.pretrained, config=self.config)

        # Store hidden size for later use
        self.hidden_size = self.config.hidden_size

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through text encoder.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (optional)

        Returns:
            Hidden states [batch_size, seq_len, hidden_size]
        """
        # Handle different model architectures
        if isinstance(self.encoder, DistilBertModel):
            # DistilBERT doesn't use token_type_ids
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            return output.last_hidden_state

        elif isinstance(self.encoder, T5EncoderModel):
            # T5 doesn't use token_type_ids
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            return output.last_hidden_state

        else:
            # Standard BERT-like models
            output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            return output.last_hidden_state

    def get_pooled_output(self,
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          token_type_ids: Optional[torch.Tensor] = None,
                          pooling_strategy: str = 'cls') -> torch.Tensor:
        """
        Get pooled representation of the input.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (optional)
            pooling_strategy: 'cls', 'mean', or 'max'

        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        hidden_states = self.forward(input_ids, attention_mask, token_type_ids)

        if pooling_strategy == 'cls':
            # Use [CLS] token representation
            return hidden_states[:, 0, :]

        elif pooling_strategy == 'mean':
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        elif pooling_strategy == 'max':
            # Max pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            return torch.max(hidden_states, 1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

    def get_attention_weights(self,
                              input_ids: torch.Tensor,
                              attention_mask: torch.Tensor,
                              token_type_ids: Optional[torch.Tensor] = None,
                              layer: int = -1) -> torch.Tensor:
        """
        Get attention weights from a specific layer.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len] (optional)
            layer: Which layer to extract attention from (-1 for last layer)

        Returns:
            Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        # Enable output_attentions for this forward pass
        original_config = self.encoder.config.output_attentions
        self.encoder.config.output_attentions = True

        try:
            if isinstance(self.encoder, DistilBertModel):
                output = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            elif isinstance(self.encoder, T5EncoderModel):
                output = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            else:
                output = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

            # Extract attention weights from specified layer
            attention_weights = output.attentions[layer]
            return attention_weights

        finally:
            # Restore original config
            self.encoder.config.output_attentions = original_config

    def freeze_layers(self, num_layers: int) -> None:
        """
        Freeze the first num_layers of the encoder.

        Args:
            num_layers: Number of layers to freeze from the beginning
        """
        if hasattr(self.encoder, 'embeddings'):
            # Freeze embeddings
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False

        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
            # Freeze specified transformer layers
            for i in range(min(num_layers, len(self.encoder.encoder.layer))):
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters in the encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the text embeddings."""
        return self.hidden_size

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the encoder."""
        return self.config.vocab_size