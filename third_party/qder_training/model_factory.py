"""
Model factory for creating QDER models from command-line arguments.
"""

from typing import Dict, Any, Type, Union
import torch.nn as nn
from qder_model import QDERModel

# Model registry
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    'qder': QDERModel,
}

# Pretrained model mappings
PRETRAINED_MODEL_MAP = {
    'bert': 'bert-base-uncased',
    'bert-large': 'bert-large-uncased',
    'distilbert': 'distilbert-base-uncased',
    'roberta': 'roberta-base',
    'roberta-large': 'roberta-large',
    'deberta': 'microsoft/deberta-base',
    'deberta-large': 'microsoft/deberta-large',
    'ernie': 'nghuyong/ernie-2.0-base-en',
    'electra': 'google/electra-small-discriminator',
    'electra-base': 'google/electra-base-discriminator',
    'conv-bert': 'YituTech/conv-bert-base',
    't5': 't5-base',
    't5-large': 't5-large'
}


def get_pretrained_model_name(text_enc: str) -> str:
    """
    Get the full pretrained model name from short name.

    Args:
        text_enc: Short name of text encoder (e.g., 'bert', 'roberta')

    Returns:
        Full pretrained model name

    Raises:
        ValueError: If text_enc is not supported
    """
    if text_enc not in PRETRAINED_MODEL_MAP:
        raise ValueError(f"Unsupported text encoder: {text_enc}. "
                         f"Supported encoders: {list(PRETRAINED_MODEL_MAP.keys())}")

    return PRETRAINED_MODEL_MAP[text_enc]


def get_model_class(model_type: str) -> Type[nn.Module]:
    """
    Get model class from model type string.

    Args:
        model_type: Type of model ('qder', 'qder_ablation')

    Returns:
        Model class

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}. "
                         f"Supported models: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[model_type]


def create_model(args) -> nn.Module:
    """
    Create model from command-line arguments.

    Args:
        args: Argument namespace with model configuration

    Returns:
        Initialized model instance
    """
    # Get pretrained model name
    pretrained = get_pretrained_model_name(args.text_enc)

    # Determine model type
    model_type = 'qder'
    if hasattr(args, 'ablation_mode') and args.ablation_mode:
        model_type = 'qder_ablation'
    elif hasattr(args, 'model_type'):
        model_type = args.model_type

    # Get model class
    model_class = get_model_class(model_type)

    # Prepare model arguments
    model_kwargs = {
        'pretrained': pretrained,
        'use_scores': getattr(args, 'use_scores', True),
        'use_entities': getattr(args, 'use_entities', True),
        'score_method': getattr(args, 'score_method', 'linear')
    }

    # Add ablation-specific arguments
    if model_type == 'qder_ablation':
        enabled_interactions = getattr(args, 'enabled_interactions', ['add', 'subtract', 'multiply'])
        model_kwargs['enabled_interactions'] = enabled_interactions

    # Create and return model
    model = model_class(**model_kwargs)

    return model


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Create model from configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized model instance
    """
    # Get model type
    model_type = config.get('model_type', 'qder')
    model_class = get_model_class(model_type)

    # Get pretrained model name
    text_enc = config.get('text_enc', 'bert')
    pretrained = get_pretrained_model_name(text_enc)
    config['pretrained'] = pretrained

    # Remove non-model arguments
    model_config = {k: v for k, v in config.items()
                    if k not in ['model_type', 'text_enc']}

    # Create and return model
    model = model_class(**model_config)

    return model


def register_model(name: str, model_class: Type[nn.Module]) -> None:
    """
    Register a new model class.

    Args:
        name: Name to register the model under
        model_class: Model class to register
    """
    MODEL_REGISTRY[name] = model_class


def list_available_models() -> Dict[str, str]:
    """
    List all available models with descriptions.

    Returns:
        Dictionary mapping model names to descriptions
    """
    descriptions = {
        'qder': 'Standard QDER model with query-document entity ranking',
        'qder_ablation': 'QDER model with configurable interaction ablations'
    }

    return {name: descriptions.get(name, 'No description available')
            for name in MODEL_REGISTRY.keys()}


def list_available_encoders() -> Dict[str, str]:
    """
    List all available text encoders.

    Returns:
        Dictionary mapping encoder names to full model names
    """
    return PRETRAINED_MODEL_MAP.copy()


def validate_model_args(args) -> None:
    """
    Validate model arguments.

    Args:
        args: Argument namespace to validate

    Raises:
        ValueError: If arguments are invalid
    """
    # Check text encoder
    if not hasattr(args, 'text_enc'):
        raise ValueError("Missing required argument: text_enc")

    if args.text_enc not in PRETRAINED_MODEL_MAP:
        raise ValueError(f"Unsupported text encoder: {args.text_enc}")

    # Check score method
    if hasattr(args, 'score_method'):
        valid_methods = ['linear', 'bilinear']
        if args.score_method not in valid_methods:
            raise ValueError(f"Invalid score_method: {args.score_method}. Must be one of {valid_methods}")

    # Check ablation interactions
    if hasattr(args, 'enabled_interactions'):
        valid_interactions = {'add', 'subtract', 'multiply'}
        for interaction in args.enabled_interactions:
            if interaction not in valid_interactions:
                raise ValueError(f"Invalid interaction: {interaction}. Must be one of {valid_interactions}")

    # Check model type
    if hasattr(args, 'model_type'):
        if args.model_type not in MODEL_REGISTRY:
            raise ValueError(f"Invalid model_type: {args.model_type}. Must be one of {list(MODEL_REGISTRY.keys())}")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive information about a model.

    Args:
        model: Model instance

    Returns:
        Dictionary with model information
    """
    info = {
        'model_class': model.__class__.__name__,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    # Add model-specific information
    if hasattr(model, 'get_model_info'):
        info.update(model.get_model_info())

    # Add text encoder information
    if hasattr(model, 'query_encoder'):
        info['text_encoder'] = {
            'pretrained': model.pretrained,
            'hidden_size': model.query_encoder.hidden_size,
            'vocab_size': model.query_encoder.get_vocab_size()
        }

    return info


def load_model_from_checkpoint(checkpoint_path: str,
                               model_args=None,
                               device: str = 'cpu') -> nn.Module:
    """
    Load model from checkpoint file.

    Args:
        checkpoint_path: Path to model checkpoint
        model_args: Optional model arguments (if not stored in checkpoint)
        device: Device to load model on

    Returns:
        Loaded model instance
    """
    import torch
    import os

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model configuration if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = create_model_from_config(config)
    elif model_args is not None:
        model = create_model(model_args)
    else:
        raise ValueError("No model configuration found in checkpoint and no model_args provided")

    # Load state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume the entire checkpoint is the state dict
        model.load_state_dict(checkpoint)

    model.to(device)
    return model


def save_model_checkpoint(model: nn.Module,
                          checkpoint_path: str,
                          additional_info: Dict[str, Any] = None) -> None:
    """
    Save model checkpoint with configuration.

    Args:
        model: Model to save
        checkpoint_path: Path to save checkpoint
        additional_info: Additional information to save (e.g., training stats)
    """
    import torch
    import os

    # Prepare checkpoint data
    checkpoint = {
        'state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }

    # Add model configuration if available
    if hasattr(model, 'config'):
        checkpoint['config'] = model.config

    # Add additional information
    if additional_info:
        checkpoint.update(additional_info)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)


def create_model_for_inference(checkpoint_path: str,
                               device: str = 'cpu') -> nn.Module:
    """
    Create model optimized for inference.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Model ready for inference
    """
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, device=device)

    # Set to evaluation mode
    model.eval()

    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False

    return model


# Example usage functions
def example_create_standard_model():
    """Example of creating a standard QDER model."""
    from argparse import Namespace

    args = Namespace(
        text_enc='bert',
        use_scores=True,
        use_entities=True,
        score_method='linear'
    )

    model = create_model(args)
    return model


def example_create_ablation_model():
    """Example of creating an ablation model."""
    from argparse import Namespace

    args = Namespace(
        text_enc='roberta',
        use_scores=True,
        use_entities=True,
        score_method='bilinear',
        ablation_mode=True,
        enabled_interactions=['add', 'multiply']  # Disable subtract
    )

    model = create_model(args)
    return model