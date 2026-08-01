import os
import time
import torch
import torch.nn as nn
import utils
import metrics
import argparse
from dataset import DocRankingDataset
from qder_model import QDERModel
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from trainer import Trainer
from torch.utils.data import DataLoader
import evaluate


def train(model, trainer, epochs, metric, qrels, valid_loader, save_path, save, run_file, eval_every, device,
          start_epoch=0, best_metric_so_far=0.0, patience=5):
    best_valid_metric = best_metric_so_far  # Start from checkpoint's best
    patience_counter = 0
    history = {'train_loss': [], 'val_metric': [], 'epoch': []}

    # Load existing history if resuming
    import json
    history_path = os.path.join(save_path, 'training_history.json')
    if start_epoch > 0 and os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            print(f'✓ Loaded training history (up to epoch {start_epoch})')
        except:
            print(f'⚠️  Could not load training history, starting fresh')

    for epoch in range(start_epoch, epochs):
        start_time = time.time()

        print(f'\n{"=" * 60}')
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'{"=" * 60}')

        # 1. Training Phase
        print('Training....')
        train_loss = trainer.train()
        history['train_loss'].append(train_loss)
        history['epoch'].append(epoch + 1)

        # 2. Validation Phase
        if (epoch + 1) % eval_every == 0:
            print('\nRunning validation...')
            res_dict = evaluate.evaluate(
                model=model,
                data_loader=valid_loader,
                device=device
            )

            # Save results with epoch number in filename
            run_filename = f"epoch_{epoch + 1:03d}_{run_file}"
            run_path = os.path.join(save_path, run_filename)
            utils.save_trec(run_path, res_dict)
            print(f'  Saved run file: {run_filename}')

            # Calculate metrics
            valid_metric = metrics.get_metric(qrels, run_path, metric)
            history['val_metric'].append(valid_metric)

            # Check for improvement
            improvement = valid_metric - best_valid_metric

            if valid_metric > best_valid_metric:
                best_valid_metric = valid_metric
                patience_counter = 0

                # Save best model checkpoint with full training state
                best_checkpoint_name = f"best_model_epoch_{epoch + 1:03d}.bin"
                best_checkpoint_path = os.path.join(save_path, best_checkpoint_name)

                # Save with full state
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_metric': best_valid_metric,
                    'config': {
                        'learning_rate': trainer.optimizer.param_groups[0]['lr'],
                        'metric': metric
                    }
                }
                torch.save(checkpoint, best_checkpoint_path)

                # Also save as the standard name for easy loading (with full state)
                standard_checkpoint_path = os.path.join(save_path, save)
                torch.save(checkpoint, standard_checkpoint_path)

                # Save a copy to "best_dev.run" for easy reference
                best_run_path = os.path.join(save_path, f"best_{run_file}")
                import shutil
                shutil.copy(run_path, best_run_path)

                print(f'\n✓ New Best {metric.upper()}: {best_valid_metric:.4f} (↑ {improvement:.4f})')
                print(f'  Saved checkpoint: {best_checkpoint_name}')
                print(f'  Saved standard checkpoint: {save}')
                print(f'  Saved best run file: best_{run_file}')
            else:
                patience_counter += 1
                print(
                    f'\n✗ No improvement. {metric.upper()}: {valid_metric:.4f} ({"↓" if improvement < 0 else "="} {abs(improvement):.4f})')
                print(f'  Patience: {patience_counter}/{patience}')

            end_time = time.time()
            epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

            print(f'\n{"─" * 60}')
            print(f'Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val {metric.upper()}: {valid_metric:.4f}')
            print(f'Best Val {metric.upper()}: {best_valid_metric:.4f}')
            print(f'{"─" * 60}')

            # Early stopping check
            if patience_counter >= patience:
                print(f'\n{"!" * 60}')
                print(f'⚠ Early stopping triggered after {epoch + 1} epochs')
                print(f'Best validation {metric.upper()}: {best_valid_metric:.4f}')
                print(f'{"!" * 60}')
                break

    # Save training history with more details
    history['best_epoch'] = history['epoch'][history['val_metric'].index(max(history['val_metric']))] if history[
        'val_metric'] else 0
    history['best_val_metric'] = max(history['val_metric']) if history['val_metric'] else 0
    history['final_train_loss'] = history['train_loss'][-1] if history['train_loss'] else 0

    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'\n✓ Training history saved to {history_path}')

    # Print summary of all saved files
    print(f'\n{"=" * 60}')
    print(f'Saved Files Summary')
    print(f'{"=" * 60}')
    print(f'Run files saved ({len(history["epoch"])} total):')
    for ep in history['epoch']:
        print(f'  - epoch_{ep:03d}_{run_file}')
    print(f'\nBest model files:')
    print(f'  - {save} (best model with full state, easy to load)')
    if history['best_epoch']:
        print(f'  - best_model_epoch_{history["best_epoch"]:03d}.bin (with epoch number)')
    print(f'  - best_{run_file} (best run file)')
    print(f'\nTraining history:')
    print(f'  - training_history.json')
    print(f'{"=" * 60}\n')

    return best_valid_metric


def main():
    parser = argparse.ArgumentParser("Script to train the QDER model.")
    parser.add_argument('--train', help='Training data (JSONL).', required=True, type=str)
    parser.add_argument('--dev', help='Development/Validation data (JSONL).', required=True, type=str)
    parser.add_argument('--qrels', help='Ground truth file in TREC format.', required=True, type=str)
    parser.add_argument('--save-dir', help='Directory where model is saved.', required=True, type=str)
    parser.add_argument('--save', help='Name of checkpoint to save.', default='model.bin', type=str)
    parser.add_argument('--checkpoint', help='Path to checkpoint to resume from.', type=str, default=None)
    parser.add_argument('--max-len', help='Max input length. Default: 512', default=512, type=int)
    parser.add_argument('--text-enc', help='Model type (bert|roberta|deberta|etc).', type=str, default='bert')
    parser.add_argument('--score-method', help='bilinear or linear.', default='bilinear',
                        choices=['linear', 'bilinear'])
    parser.add_argument('--epoch', help='Number of epochs. Default: 20', type=int, default=20)
    parser.add_argument('--patience', help='Early stopping patience (epochs). Default: 5',
                        type=int, default=5)
    parser.add_argument('--batch-size', help='Size of each batch. Default: 8.', type=int, default=8)
    parser.add_argument('--learning-rate', help='Learning rate. Default: 1e-5.', type=float, default=1e-5)
    parser.add_argument('--n-warmup-steps', help='Warmup steps for scheduler. Default: 1000.', type=int, default=1000)
    parser.add_argument('--metric', help='Metric for validation (map|ndcg). Default: map', default='map', type=str)
    parser.add_argument('--eval-every', help='Evaluate every X epochs.', type=int, default=1)
    parser.add_argument('--num-workers', help='DataLoader workers.', type=int, default=0)
    parser.add_argument('--cuda', help='CUDA device index.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Use CUDA if available.', action='store_true')
    parser.add_argument('--use-entities', help='Train with the entity channel. Off by default: '
                        'trecrag has no real WAT/Wikipedia2Vec entity linker for ClimbMix, and a '
                        'checkpoint trained with this on cannot load into trecrag\'s default '
                        'use_entities: false inference config (shape mismatch).',
                        action='store_true')
    parser.add_argument('--run', help='Output dev run file name.', default='dev.run', type=str)
    parser.add_argument('--enabled-interactions', help='Enabled interactions (add,multiply,subtract)',
                        type=str, default='add,multiply')

    args = parser.parse_args()

    # Environment Setup
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f'\n{"=" * 60}')
    print(f'QDER Training Configuration')
    print(f'{"=" * 60}')
    print(f'Device: {device}')
    print(f'Learning Rate: {args.learning_rate}')
    print(f'Batch Size: {args.batch_size}')
    print(f'Epochs: {args.epoch}')
    print(f'Enabled Interactions: {args.enabled_interactions}')
    if args.checkpoint:
        print(f'Resume from: {args.checkpoint}')
    print(f'{"=" * 60}\n')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model_map = {
        'bert': 'bert-base-uncased',
        'roberta': 'roberta-base',
        'deberta': 'microsoft/deberta-base'
    }
    pretrain = model_map.get(args.text_enc, 'bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(pretrain, model_max_length=args.max_len)

    # Data Loading
    print('Creating datasets...')
    train_set = DocRankingDataset(dataset=args.train, tokenizer=tokenizer, train=True, max_len=args.max_len)
    dev_set = DocRankingDataset(dataset=args.dev, tokenizer=tokenizer, train=False, max_len=args.max_len)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_set.collate
    )
    dev_loader = DataLoader(
        dataset=dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dev_set.collate
    )

    # Model Initialization
    enabled_interactions = [i.strip() for i in args.enabled_interactions.split(',')]

    model = QDERModel(
        pretrained=pretrain,
        use_scores=True,
        use_entities=args.use_entities,
        score_method=args.score_method,
        enabled_interactions=enabled_interactions
    )

    # Initialize variables for checkpoint resuming
    start_epoch = 0
    best_metric_so_far = 0.0
    optimizer_state = None
    scheduler_state = None

    # Load checkpoint if provided
    if args.checkpoint:
        print(f'\n{"=" * 60}')
        print(f'Loading checkpoint from: {args.checkpoint}')
        print(f'{"=" * 60}')

        if not os.path.exists(args.checkpoint):
            print(f'⚠️  Checkpoint file not found: {args.checkpoint}')
            print(f'   Starting training from scratch...')
        else:
            try:
                checkpoint = torch.load(args.checkpoint, map_location='cpu')

                # Load model weights
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f'✓ Loaded model weights')

                    # Store optimizer and scheduler states
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer_state = checkpoint['optimizer_state_dict']
                        print(f'✓ Found optimizer state')

                    if 'scheduler_state_dict' in checkpoint:
                        scheduler_state = checkpoint['scheduler_state_dict']
                        print(f'✓ Found scheduler state')

                    # Load training progress
                    start_epoch = checkpoint.get('epoch', 0)
                    best_metric_so_far = checkpoint.get('best_metric', 0.0)

                    if 'config' in checkpoint:
                        print(f'✓ Checkpoint config:')
                        for key, value in checkpoint['config'].items():
                            print(f'    {key}: {value}')

                    print(f'✓ Resuming from epoch {start_epoch}')
                    print(f'✓ Best metric so far: {best_metric_so_far:.4f}')
                else:
                    # Assume it's just state_dict (older format)
                    model.load_state_dict(checkpoint)
                    print(f'✓ Loaded model weights (legacy format, no training state)')

                print(f'✓ Checkpoint loaded successfully')
                print(f'{"=" * 60}\n')

            except Exception as e:
                print(f'⚠️  Error loading checkpoint: {e}')
                print(f'   Starting training from scratch...')
                import traceback
                traceback.print_exc()

    # Move model to device
    model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )

    # Load optimizer state if available
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print(f'✓ Loaded optimizer state')
        except Exception as e:
            print(f'⚠️  Could not load optimizer state: {e}')

    # Linear scheduling with warmup
    num_training_steps = (len(train_set) // args.batch_size) * args.epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.n_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Load scheduler state if available
    if scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
            print(f'✓ Loaded scheduler state')
        except Exception as e:
            print(f'⚠️  Could not load scheduler state: {e}')

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if start_epoch > 0:
        print(f'\n{"=" * 60}')
        print(f'Resuming Training from Epoch {start_epoch + 1}')
        print(f'{"=" * 60}')
    else:
        print(f'\n{"=" * 60}')
        print(f'Starting Training')
        print(f'{"=" * 60}')

    # Sanity check
    print('\nRunning sanity check...')
    try:
        test_batch = next(iter(train_loader))
        with torch.no_grad():
            test_output = model(
                query_input_ids=test_batch['query_input_ids'].to(device),
                query_attention_mask=test_batch['query_attention_mask'].to(device),
                query_token_type_ids=test_batch['query_token_type_ids'].to(device),
                query_entity_emb=test_batch['query_entity_emb'].to(device),
                doc_input_ids=test_batch['doc_input_ids'].to(device),
                doc_attention_mask=test_batch['doc_attention_mask'].to(device),
                doc_token_type_ids=test_batch['doc_token_type_ids'].to(device),
                doc_entity_emb=test_batch['doc_entity_emb'].to(device),
                query_entity_mask=test_batch['query_entity_mask'].to(device),
                doc_entity_mask=test_batch['doc_entity_mask'].to(device),
                doc_score=test_batch['doc_score'].to(device)
            )
        print(f'✓ Forward pass successful')
        print(f'  Score range: [{test_output["score"].min():.2f}, {test_output["score"].max():.2f}]')
        print(f'  Score mean: {test_output["score"].mean():.2f}')
    except Exception as e:
        print(f'✗ Sanity check failed: {e}')
        import traceback
        traceback.print_exc()
        return

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=loss_fn,
        scheduler=scheduler,
        metric=args.metric,
        data_loader=train_loader,
        use_cuda=args.use_cuda,
        device=device
    )

    best_metric = train(
        model=model,
        trainer=trainer,
        epochs=args.epoch,
        metric=args.metric,
        qrels=args.qrels,
        valid_loader=dev_loader,
        save_path=args.save_dir,
        save=args.save,
        run_file=args.run,
        eval_every=args.eval_every,
        device=device,
        start_epoch=start_epoch,
        best_metric_so_far=best_metric_so_far,
        patience=args.patience
    )

    print(f'\n{"=" * 60}')
    print(f'Training Complete')
    print(f'{"=" * 60}')
    print(f'Best Validation {args.metric.upper()}: {best_metric:.4f}')
    print(f'Model saved to: {os.path.join(args.save_dir, args.save)}')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()