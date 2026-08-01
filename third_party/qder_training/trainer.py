import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, metric, data_loader, use_cuda, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.metric = metric
        self.data_loader = data_loader
        self.use_cuda = use_cuda
        self.device = device

    def train(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        num_nan_batches = 0

        for batch in tqdm(self.data_loader, desc="Training"):
            self.optimizer.zero_grad()

            # Forward pass with corrected QDER signature
            output = self.model(
                query_input_ids=batch['query_input_ids'].to(self.device),
                query_attention_mask=batch['query_attention_mask'].to(self.device),
                query_token_type_ids=batch['query_token_type_ids'].to(self.device),
                query_entity_emb=batch['query_entity_emb'].to(self.device),
                doc_input_ids=batch['doc_input_ids'].to(self.device),
                doc_attention_mask=batch['doc_attention_mask'].to(self.device),
                doc_token_type_ids=batch['doc_token_type_ids'].to(self.device),
                doc_entity_emb=batch['doc_entity_emb'].to(self.device),
                query_entity_mask=batch['query_entity_mask'].to(self.device),
                doc_entity_mask=batch['doc_entity_mask'].to(self.device),
                doc_score=batch['doc_score'].to(self.device)
            )

            logits = output['score']

            # Check for NaN/Inf BEFORE loss calculation
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"\n⚠ Warning: NaN/Inf detected in logits, skipping batch")
                num_nan_batches += 1
                continue

            loss = self.criterion(logits, batch['label'].to(self.device))

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"\n⚠ Warning: NaN loss detected, skipping batch")
                num_nan_batches += 1
                continue

            loss.backward()

            # **CRITICAL: Gradient clipping to prevent exploding gradients**
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        if num_nan_batches > 0:
            print(f"\n⚠ Total batches with NaN/Inf: {num_nan_batches}/{len(self.data_loader)}")

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss