from typing import Dict, Any
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


class DocRankingDataset(Dataset):
    def __init__(self, dataset, tokenizer, train, max_len):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._train = train
        self._read_data()
        self._count = len(self._examples)

    def __len__(self):
        return self._count

    def __getitem__(self, idx):
        example: Dict[str, Any] = self._examples[idx]
        query_input_ids, query_attention_mask, query_token_type_ids = self._create_input(
            text=example['query']
        )
        doc_input_ids, doc_attention_mask, doc_token_type_ids = self._create_input(
            text=example['doc']
        )
        doc_entity_emb = example['doc_ent_emb']
        query_entity_emb = example['query_ent_emb']
        doc_score = example['doc_score']

        if self._train:
            return {
                'query_input_ids': query_input_ids,
                'query_attention_mask': query_attention_mask,
                'query_token_type_ids': query_token_type_ids,
                'query_entity_emb': query_entity_emb,
                'doc_input_ids': doc_input_ids,
                'doc_attention_mask': doc_attention_mask,
                'doc_token_type_ids': doc_token_type_ids,
                'doc_entity_emb': doc_entity_emb,
                'doc_score': doc_score,
                'label': example['label']
            }
        else:
            return {
                'query_id': example['query_id'],
                'doc_id': example['doc_id'],
                'label': example['label'],
                'query_input_ids': query_input_ids,
                'query_attention_mask': query_attention_mask,
                'query_token_type_ids': query_token_type_ids,
                'query_entity_emb': query_entity_emb,
                'doc_input_ids': doc_input_ids,
                'doc_attention_mask': doc_attention_mask,
                'doc_token_type_ids': doc_token_type_ids,
                'doc_entity_emb': doc_entity_emb,
                'doc_score': doc_score,
            }

    def _create_input(self, text):
        # Use __call__ instead of deprecated encode_plus
        encoded_dict = self._tokenizer(
            text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self._max_len,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_token_type_ids=True,  # Construct token type ids
            return_tensors=None  # Return as lists, not tensors
        )
        # Ensure we return lists for consistency with collate function
        return (
            encoded_dict['input_ids'],
            encoded_dict['attention_mask'],
            encoded_dict['token_type_ids']
        )

    def _read_data(self):
        with open(self._dataset, 'r') as f:
            self._examples = [json.loads(line) for line in tqdm(f)]

    def collate(self, batch):
        query_input_ids = torch.tensor([item['query_input_ids'] for item in batch])
        query_attention_mask = torch.tensor([item['query_attention_mask'] for item in batch])
        query_token_type_ids = torch.tensor([item['query_token_type_ids'] for item in batch])
        doc_input_ids = torch.tensor([item['doc_input_ids'] for item in batch])
        doc_attention_mask = torch.tensor([item['doc_attention_mask'] for item in batch])
        doc_token_type_ids = torch.tensor([item['doc_token_type_ids'] for item in batch])
        doc_score = torch.tensor([item['doc_score'] for item in batch])
        label = torch.tensor([item['label'] for item in batch], dtype=torch.float)

        # Pad entity embeddings to handle variable-length sequences
        query_entity_emb_list = [item['query_entity_emb'] for item in batch]
        doc_entity_emb_list = [item['doc_entity_emb'] for item in batch]

        # Get entity lengths before padding
        query_entity_lengths = [len(emb) for emb in query_entity_emb_list]
        doc_entity_lengths = [len(emb) for emb in doc_entity_emb_list]

        # Handle empty entity sequences safely
        if any(len(emb) > 0 for emb in query_entity_emb_list):
            query_entity_emb = pad_sequence(
                [torch.as_tensor(b, dtype=torch.float) for b in query_entity_emb_list if len(b) > 0],
                batch_first=True
            )
            # If some samples have no entities, pad them with zeros
            if len(query_entity_emb) < len(batch):
                embedding_dim = query_entity_emb.shape[-1]
                max_len = query_entity_emb.shape[1]
                full_query_emb = torch.zeros(len(batch), max_len, embedding_dim)
                idx = 0
                for i, emb in enumerate(query_entity_emb_list):
                    if len(emb) > 0:
                        full_query_emb[i] = query_entity_emb[idx]
                        idx += 1
                query_entity_emb = full_query_emb
        else:
            # All samples have no query entities - create dummy tensor
            query_entity_emb = torch.zeros(len(batch), 1, 300)  # Assuming 300-dim embeddings
            query_entity_lengths = [0] * len(batch)

        if any(len(emb) > 0 for emb in doc_entity_emb_list):
            doc_entity_emb = pad_sequence(
                [torch.as_tensor(b, dtype=torch.float) for b in doc_entity_emb_list if len(b) > 0],
                batch_first=True
            )
            # If some samples have no entities, pad them with zeros
            if len(doc_entity_emb) < len(batch):
                embedding_dim = doc_entity_emb.shape[-1]
                max_len = doc_entity_emb.shape[1]
                full_doc_emb = torch.zeros(len(batch), max_len, embedding_dim)
                idx = 0
                for i, emb in enumerate(doc_entity_emb_list):
                    if len(emb) > 0:
                        full_doc_emb[i] = doc_entity_emb[idx]
                        idx += 1
                doc_entity_emb = full_doc_emb
        else:
            # All samples have no doc entities - create dummy tensor
            doc_entity_emb = torch.zeros(len(batch), 1, 300)  # Assuming 300-dim embeddings
            doc_entity_lengths = [0] * len(batch)

        max_query_entity_len = query_entity_emb.shape[1]
        max_doc_entity_len = doc_entity_emb.shape[1]

        # Create boolean masks: True for actual elements, False for padding
        query_entity_mask = torch.zeros(len(batch), max_query_entity_len, dtype=torch.bool)
        doc_entity_mask = torch.zeros(len(batch), max_doc_entity_len, dtype=torch.bool)

        for i, (q_len, d_len) in enumerate(zip(query_entity_lengths, doc_entity_lengths)):
            if q_len > 0:
                query_entity_mask[i, :q_len] = True
            if d_len > 0:
                doc_entity_mask[i, :d_len] = True

        if self._train:
            return {
                'query_input_ids': query_input_ids,
                'query_attention_mask': query_attention_mask,
                'query_token_type_ids': query_token_type_ids,
                'query_entity_emb': query_entity_emb,
                'query_entity_mask': query_entity_mask,
                'doc_input_ids': doc_input_ids,
                'doc_attention_mask': doc_attention_mask,
                'doc_token_type_ids': doc_token_type_ids,
                'doc_entity_emb': doc_entity_emb,
                'doc_entity_mask': doc_entity_mask,
                'doc_score': doc_score,
                'label': label
            }
        else:
            query_id = [item['query_id'] for item in batch]
            doc_id = [item['doc_id'] for item in batch]
            return {
                'query_id': query_id,
                'doc_id': doc_id,
                'label': label,
                'query_input_ids': query_input_ids,
                'query_attention_mask': query_attention_mask,
                'query_token_type_ids': query_token_type_ids,
                'query_entity_emb': query_entity_emb,
                'query_entity_mask': query_entity_mask,
                'doc_input_ids': doc_input_ids,
                'doc_attention_mask': doc_attention_mask,
                'doc_token_type_ids': doc_token_type_ids,
                'doc_entity_emb': doc_entity_emb,
                'doc_entity_mask': doc_entity_mask,
                'doc_score': doc_score,
            }