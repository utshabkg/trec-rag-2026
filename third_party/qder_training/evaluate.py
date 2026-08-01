import torch
from tqdm import tqdm


def evaluate(model, data_loader, device):
    """
    Performs inference on the test/validation set.
    Returns a result dictionary {query_id: {doc_id: [score, label]}}.
    """
    rst_dict = {}
    model.eval()  # Ensure dropout/batchnorm are in eval mode

    num_batch = len(data_loader)

    with torch.no_grad():  # Disable gradient tracking for efficiency
        for batch in tqdm(data_loader, total=num_batch, desc="Evaluating"):
            # Extract metadata needed for TREC formatting
            query_ids, doc_ids, labels = batch['query_id'], batch['doc_id'], batch['label']

            # Forward pass with corrected QDER signature
            output = model(
                query_input_ids=batch['query_input_ids'].to(device),
                query_attention_mask=batch['query_attention_mask'].to(device),
                query_token_type_ids=batch['query_token_type_ids'].to(device),
                query_entity_emb=batch['query_entity_emb'].to(device),
                doc_input_ids=batch['doc_input_ids'].to(device),
                doc_attention_mask=batch['doc_attention_mask'].to(device),
                doc_token_type_ids=batch['doc_token_type_ids'].to(device),
                doc_entity_emb=batch['doc_entity_emb'].to(device),
                query_entity_mask=batch['query_entity_mask'].to(device),
                doc_entity_mask=batch['doc_entity_mask'].to(device),
                doc_score=batch['doc_score'].to(device)
            )

            batch_score = output['score']

            # Convert scores to list and store in the result dictionary
            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, score, l) in zip(query_ids, doc_ids, batch_score, labels):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}

                # Store the score and original label (for metric calculation)
                # Take max score if multiple chunks exist for one doc
                if d_id not in rst_dict[q_id] or score > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [score, l]

    return rst_dict