"""
QDER model with proper implementation matching the paper.

This is the corrected implementation that:
- Uses dynamic dimensions based on enabled interactions
- Applies proper attention and pooling masking
- Only concatenates enabled interaction features
- Matches the paper's "No-Subtract" (add + multiply) configuration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from base_model import BaseQDERModel
from text_embedding import TextEmbedding


class QDERModel(BaseQDERModel):
    """
    Query-Specific Document and Entity Representations (QDER) model.

    This implementation properly matches the paper by using only Addition and
    Multiplication interactions (the "No-Subtract" configuration from Table 4).

    Key features:
    - Dynamic dimension calculation based on enabled interactions
    - Proper attention masking for padding tokens and entities
    - Masked mean pooling for accurate aggregation
    - Flexible ablation support for interaction studies
    """

    def __init__(self,
                 pretrained: str,
                 use_scores: bool = True,
                 use_entities: bool = True,
                 score_method: str = 'bilinear',
                 enabled_interactions: List[str] = None) -> None:
        """
        Initialize QDER model.

        Args:
            pretrained: Name of pretrained text encoder (e.g., 'bert-base-uncased')
            use_scores: Whether to use document retrieval scores for weighting
            use_entities: Whether to use entity embeddings
            score_method: Scoring method ('linear' or 'bilinear')
            enabled_interactions: List of enabled interactions.
                                Default: ['add', 'multiply'] (paper's best config)
        """
        # Initialize base
        super(QDERModel, self).__init__(pretrained, use_scores, use_entities, score_method)

        # Set default to paper's best configuration: Addition + Multiplication only
        if enabled_interactions is None:
            enabled_interactions = ['add', 'multiply']

        # Validate interactions
        valid_interactions = {'add', 'subtract', 'multiply'}
        for interaction in enabled_interactions:
            if interaction not in valid_interactions:
                raise ValueError(
                    f"Invalid interaction: {interaction}. "
                    f"Must be one of {valid_interactions}"
                )

        self.enabled_interactions = enabled_interactions

        # Update config
        self.config['enabled_interactions'] = enabled_interactions

        # 1. Initialize text encoders
        self.query_encoder = TextEmbedding(pretrained=pretrained)
        self.doc_encoder = TextEmbedding(pretrained=pretrained)

        # 2. Dynamic dimension calculation based on ENABLED interactions
        text_dim = self.query_encoder.get_embedding_dim()  # 768 for BERT
        entity_dim = 300  # Entity embedding dimension

        num_interactions = len(self.enabled_interactions)

        # Calculate total dimension based ONLY on enabled interactions
        total_dim = num_interactions * text_dim
        if use_entities:
            total_dim += num_interactions * entity_dim

        # 3. Initialize scoring layer with correct dimensions
        self.layer_norm = nn.LayerNorm(total_dim)
        if score_method == 'linear':
            self.score = nn.Linear(in_features=total_dim, out_features=1)
        elif score_method == 'bilinear':
            self.score = nn.Bilinear(
                in1_features=total_dim,
                in2_features=total_dim,
                out_features=1
            )
        else:
            raise ValueError(f"Unknown score_method: {score_method}")

    def get_text_embeddings(self,
                            input_ids: torch.Tensor,
                            attention_mask: torch.Tensor,
                            token_type_ids: torch.Tensor,
                            encoder_type: str = 'query') -> torch.Tensor:
        """
        Get text embeddings from the specified encoder.

        Required implementation of abstract method from BaseQDERModel.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            encoder_type: 'query' or 'doc' encoder

        Returns:
            Text embeddings [batch_size, seq_len, hidden_size]
        """
        if encoder_type == 'query':
            return self.query_encoder(input_ids, attention_mask, token_type_ids)
        elif encoder_type == 'doc':
            return self.doc_encoder(input_ids, attention_mask, token_type_ids)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

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
        Forward pass through QDER model.

        Args:
            query_input_ids: Query token IDs [batch_size, seq_len]
            query_attention_mask: Query attention mask [batch_size, seq_len]
            query_token_type_ids: Query token type IDs [batch_size, seq_len]
            query_entity_emb: Query entity embeddings [batch_size, num_q_entities, 300]
            doc_input_ids: Document token IDs [batch_size, seq_len]
            doc_attention_mask: Document attention mask [batch_size, seq_len]
            doc_token_type_ids: Document token type IDs [batch_size, seq_len]
            doc_entity_emb: Document entity embeddings [batch_size, num_d_entities, 300]
            query_entity_mask: Query entity attention mask [batch_size, num_q_entities]
            doc_entity_mask: Document entity attention mask [batch_size, num_d_entities]
            doc_score: Document retrieval scores [batch_size]

        Returns:
            Dictionary containing 'score' and 'combined_emb'
        """
        # ===== 1. GET TEXT EMBEDDINGS =====
        query_text_emb = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            token_type_ids=query_token_type_ids
        )  # [batch, q_len, hidden_dim]

        doc_text_emb = self.doc_encoder(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
            token_type_ids=doc_token_type_ids
        )  # [batch, d_len, hidden_dim]

        # ===== 2. TEXT-BASED ATTENTION (WITH MASKING) =====
        # Compute attention scores
        text_attention_scores = torch.matmul(
            query_text_emb,
            doc_text_emb.transpose(-2, -1)
        )  # [batch, q_len, d_len]

        # Apply document attention mask to prevent attending to padding
        if doc_attention_mask is not None:
            doc_mask_expanded = doc_attention_mask.unsqueeze(1)  # [batch, 1, d_len]
            text_attention_scores = text_attention_scores.masked_fill(
                doc_mask_expanded == 0,
                float('-inf')
            )

        text_attention_probs = F.softmax(text_attention_scores, dim=-1)
        weighted_doc_text_emb = torch.matmul(text_attention_probs, doc_text_emb)
        # [batch, q_len, hidden_dim]

        # ===== 3. TEXT INTERACTION COMPUTATION =====
        text_interactions = []

        if 'subtract' in self.enabled_interactions:
            text_interactions.append(
                torch.sub(query_text_emb, weighted_doc_text_emb, alpha=1)
            )
        if 'add' in self.enabled_interactions:
            text_interactions.append(
                torch.add(query_text_emb, weighted_doc_text_emb, alpha=1)
            )
        if 'multiply' in self.enabled_interactions:
            text_interactions.append(
                query_text_emb * weighted_doc_text_emb
            )

        # ===== 4. TEXT POOLING (WITH MASKING) =====
        query_mask_expanded = query_attention_mask.unsqueeze(-1).float()
        # [batch, q_len, 1]
        sum_mask = torch.clamp(query_mask_expanded.sum(dim=1), min=1e-9)
        # [batch, 1]

        pooled_text_features = []
        for interaction in text_interactions:
            # Apply mask: zero out padding positions
            masked_interaction = interaction * query_mask_expanded

            # Compute mean over non-padded positions
            pooled = torch.sum(masked_interaction, dim=1) / sum_mask
            # [batch, hidden_dim]

            # Apply document score weighting if enabled
            if self.use_scores and doc_score is not None:
                pooled = pooled * doc_score.unsqueeze(-1)

            pooled_text_features.append(pooled)

        # ===== 5. ENTITY-BASED INTERACTIONS (WITH MASKING) =====
        pooled_entity_features = []

        if self.use_entities and query_entity_emb.size(1) > 0 and doc_entity_emb.size(1) > 0:
            # Entity attention
            entity_attention_scores = torch.matmul(
                query_entity_emb,
                doc_entity_emb.transpose(-2, -1)
            )  # [batch, num_q_ent, num_d_ent]

            # Apply document entity mask
            if doc_entity_mask is not None:
                doc_entity_mask_expanded = doc_entity_mask.unsqueeze(1)
                # [batch, 1, num_d_ent]
                entity_attention_scores = entity_attention_scores.masked_fill(
                    ~doc_entity_mask_expanded,
                    float('-inf')
                )

            entity_attention_probs = F.softmax(entity_attention_scores, dim=-1)
            # Handle NaN from all-padding rows
            entity_attention_probs = torch.nan_to_num(entity_attention_probs, nan=0.0)

            weighted_doc_entity_emb = torch.matmul(
                entity_attention_probs,
                doc_entity_emb
            )  # [batch, num_q_ent, 300]

            # Compute entity interactions
            entity_interactions = []

            if 'subtract' in self.enabled_interactions:
                entity_interactions.append(
                    torch.sub(query_entity_emb, weighted_doc_entity_emb, alpha=1)
                )
            if 'add' in self.enabled_interactions:
                entity_interactions.append(
                    torch.add(query_entity_emb, weighted_doc_entity_emb, alpha=1)
                )
            if 'multiply' in self.enabled_interactions:
                entity_interactions.append(
                    query_entity_emb * weighted_doc_entity_emb
                )

            # Entity pooling with masking
            if query_entity_mask is not None:
                query_entity_mask_expanded = query_entity_mask.unsqueeze(-1).float()
                # [batch, num_q_ent, 1]
                entity_sum_mask = torch.clamp(
                    query_entity_mask_expanded.sum(dim=1),
                    min=1e-9
                )  # [batch, 1]

                for interaction in entity_interactions:
                    # Apply mask
                    masked = interaction * query_entity_mask_expanded

                    # Compute mean over non-padded entities
                    pooled = torch.sum(masked, dim=1) / entity_sum_mask
                    # [batch, 300]

                    # Apply document score weighting if enabled
                    if self.use_scores and doc_score is not None:
                        pooled = pooled * doc_score.unsqueeze(-1)

                    pooled_entity_features.append(pooled)
            else:
                # Fallback: simple mean pooling without masking
                for interaction in entity_interactions:
                    pooled = torch.mean(interaction, dim=1)

                    if self.use_scores and doc_score is not None:
                        pooled = pooled * doc_score.unsqueeze(-1)

                    pooled_entity_features.append(pooled)

        elif self.use_entities:
            # Entities enabled but not present in batch: provide zero features
            batch_size = query_text_emb.size(0)
            device = query_text_emb.device
            entity_dim = 300

            for _ in range(len(self.enabled_interactions)):
                pooled_entity_features.append(
                    torch.zeros(batch_size, entity_dim, device=device)
                )

        # ===== 6. CONCATENATION =====
        # Concatenate only the computed (enabled) features
        all_features = pooled_text_features + pooled_entity_features
        combined_emb = torch.cat(all_features, dim=-1)
        combined_emb = self.layer_norm(combined_emb)

        # ===== 7. SCORING =====
        if self.score_method == 'linear':
            score = self.score(combined_emb)
        else:  # bilinear
            score = self.score(combined_emb, combined_emb)

        return {
            'score': score.squeeze(-1),
            'combined_emb': combined_emb
        }