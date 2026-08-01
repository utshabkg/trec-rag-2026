import os
import torch
import argparse
from dataset import DocRankingDataset
from models import QDERModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import utils
import metrics
from tqdm import tqdm


def evaluate(model, data_loader, device):
    """
    Performs inference on the test/validation set.
    Returns a result dictionary {query_id: {doc_id: [score, label]}}.
    """
    rst_dict = {}
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            query_ids, doc_ids, labels = batch['query_id'], batch['doc_id'], batch['label']

            # Forward pass
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
            batch_score = batch_score.detach().cpu().tolist()

            for (q_id, d_id, score, l) in zip(query_ids, doc_ids, batch_score, labels):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}

                # Take max score if multiple instances exist
                if d_id not in rst_dict[q_id] or score > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [score, l]

    return rst_dict


def main():
    parser = argparse.ArgumentParser("Script to test the QDER model.")
    parser.add_argument('--test', help='Test data (JSONL).', required=True, type=str)
    parser.add_argument('--qrels', help='Ground truth file in TREC format.', required=True, type=str)
    parser.add_argument('--checkpoint', help='Path to model checkpoint.', required=True, type=str)
    parser.add_argument('--output', help='Output run file.', default='test.run', type=str)
    parser.add_argument('--max-len', help='Max input length.', default=512, type=int)
    parser.add_argument('--text-enc', help='Model type (bert|roberta|deberta).', type=str, default='bert')
    parser.add_argument('--score-method', help='Scoring method.', default='bilinear',
                        choices=['linear', 'bilinear'])
    parser.add_argument('--batch-size', help='Batch size.', type=int, default=16)
    parser.add_argument('--num-workers', help='DataLoader workers.', type=int, default=0)
    parser.add_argument('--cuda', help='CUDA device index.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Use CUDA if available.', action='store_true')
    parser.add_argument('--enabled-interactions', help='Enabled interactions (add,multiply,subtract)',
                        type=str, default='add,multiply')
    parser.add_argument('--metrics', help='Metrics to report (comma-separated).',
                        type=str, default='map,ndcg')

    args = parser.parse_args()

    # Setup device
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")

    # Model configuration
    model_map = {
        'bert': 'bert-base-uncased',
        'roberta': 'roberta-base',
        'deberta': 'microsoft/deberta-base'
    }
    pretrain = model_map.get(args.text_enc, 'bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(pretrain, model_max_length=args.max_len)

    # Load test data
    print('Loading test dataset...')
    test_set = DocRankingDataset(
        dataset=args.test,
        tokenizer=tokenizer,
        train=False,  # Test mode
        max_len=args.max_len
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_set.collate
    )

    # Initialize model
    print('Initializing model...')
    enabled_interactions = [i.strip() for i in args.enabled_interactions.split(',')]

    model = QDERModel(
        pretrained=pretrain,
        use_scores=True,
        use_entities=True,
        score_method=args.score_method,
        enabled_interactions=enabled_interactions
    )

    # Load checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    print(f"Model loaded successfully")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run evaluation
    print('Running evaluation...')
    results = evaluate(model, test_loader, device)

    # Save run file
    print(f'Saving results to {args.output}...')
    utils.save_trec(args.output, results)

    # Calculate and report metrics
    print('\nResults:')
    print('-' * 50)

    metric_list = [m.strip() for m in args.metrics.split(',')]
    for metric_name in metric_list:
        try:
            metric_value = metrics.get_metric(args.qrels, args.output, metric_name)
            print(f'{metric_name.upper()}: {metric_value:.4f}')
        except Exception as e:
            print(f'Could not compute {metric_name}: {e}')

    print('-' * 50)
    print('Evaluation complete!')


if __name__ == '__main__':
    main()