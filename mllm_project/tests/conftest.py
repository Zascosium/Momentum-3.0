"""
Pytest configuration and fixtures for the MLLM test suite.

This module provides common fixtures and configuration for all tests.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

# Add project source to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import project modules
from utils.config_loader import load_config_for_training
from models.multimodal_model import MultimodalLLM
from data.preprocessing import TimeSeriesPreprocessor, TextPreprocessor


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return {
        'time_series': {
            'max_length': 256,
            'n_features': 3,
            'normalization': 'standard',
            'preprocessing': {
                'fill_missing': 'interpolate',
                'outlier_detection': True,
                'outlier_threshold': 3.0
            }
        },
        'text': {
            'tokenizer': {
                'model_name': 'gpt2',
                'max_length': 128,
                'padding': True,
                'truncation': True
            },
            'preprocessing': {
                'lowercase': True,
                'remove_special_chars': False,
                'min_length': 5
            }
        },
        'model': {
            'moment_encoder': {
                'model_name': 'AutonLab/MOMENT-1-large',
                'freeze_encoder': False,
                'output_dim': 512
            },
            'text_decoder': {
                'model_name': 'gpt2-medium',
                'freeze_embeddings': False,
                'freeze_layers': 0
            },
            'projection': {
                'hidden_dim': 1024,
                'dropout': 0.1,
                'num_layers': 2,
                'activation': 'gelu'
            },
            'cross_attention': {
                'num_heads': 8,
                'hidden_dim': 512,
                'dropout': 0.1,
                'num_layers': 4
            },
            'fusion': {
                'strategy': 'cross_attention',
                'temperature': 1.0
            }
        },
        'training': {
            'batch_size': 8,
            'learning_rate': 1e-4,
            'num_epochs': 5,
            'warmup_steps': 100,
            'gradient_clip': 1.0,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'mixed_precision': True
        },
        'data': {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'seed': 42,
            'num_workers': 4,
            'pin_memory': True
        },
        'metrics': {
            'primary': 'loss',
            'track_perplexity': True,
            'track_bleu': True,
            'track_rouge': True,
            'custom_metrics': []
        }
    }


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Provide device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Provide temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_time_series() -> np.ndarray:
    """Generate sample time series data."""
    np.random.seed(42)
    seq_len = 256
    n_features = 3
    
    # Create realistic time series with trends and seasonality
    t = np.linspace(0, 10, seq_len)
    
    # Feature 1: Trend with noise
    feature1 = 2 * t + 0.5 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.2, seq_len)
    
    # Feature 2: Seasonal pattern
    feature2 = 5 + 3 * np.sin(2 * np.pi * t / 2) + np.random.normal(0, 0.3, seq_len)
    
    # Feature 3: Random walk
    feature3 = np.cumsum(np.random.normal(0, 0.1, seq_len))
    
    return np.column_stack([feature1, feature2, feature3])


@pytest.fixture
def sample_text_data() -> list[str]:
    """Generate sample text data."""
    return [
        "The time series shows an upward trend with seasonal variations.",
        "Market volatility increased significantly during this period.",
        "Temperature readings indicate a warming pattern over time.",
        "Energy consumption peaks during winter months.",
        "Financial data reveals cyclical patterns in trading volume."
    ]


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode.return_value = torch.randint(0, 1000, (10,))
    tokenizer.decode.return_value = "Sample decoded text"
    tokenizer.vocab_size = 50257
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.bos_token_id = 2
    return tokenizer


@pytest.fixture
def sample_batch(sample_time_series, sample_text_data, test_config):
    """Generate sample batch for testing."""
    batch_size = test_config['training']['batch_size']
    seq_len = test_config['time_series']['max_length']
    text_len = test_config['text']['tokenizer']['max_length']
    n_features = test_config['time_series']['n_features']
    
    # Create batch data
    time_series = torch.randn(batch_size, seq_len, n_features)
    ts_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    text_input_ids = torch.randint(0, 1000, (batch_size, text_len))
    text_attention_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    
    return {
        'time_series': time_series,
        'ts_attention_mask': ts_attention_mask,
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'domains': ['test'] * batch_size
    }


@pytest.fixture
def mock_model(test_config, device):
    """Create mock model for testing."""
    with patch('transformers.AutoModel.from_pretrained'), \
         patch('transformers.AutoTokenizer.from_pretrained'), \
         patch('transformers.GPT2LMHeadModel.from_pretrained'):
        
        model = MultimodalLLM(test_config)
        model.to(device)
        return model


@pytest.fixture
def time_series_preprocessor(test_config):
    """Create time series preprocessor."""
    return TimeSeriesPreprocessor(test_config['time_series'])


@pytest.fixture
def text_preprocessor(test_config):
    """Create text preprocessor."""
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        # Configure mock tokenizer
        mock_instance = Mock()
        mock_instance.encode.return_value = [1, 2, 3, 4, 5]
        mock_instance.decode.return_value = "test text"
        mock_instance.vocab_size = 50257
        mock_instance.pad_token_id = 0
        mock_instance.eos_token_id = 1
        mock_tokenizer.return_value = mock_instance
        
        return TextPreprocessor(test_config['text'])


# Test data generators
def generate_synthetic_dataset(num_samples: int = 100, config: Dict[str, Any] = None):
    """Generate synthetic dataset for testing."""
    if config is None:
        config = {
            'time_series': {'max_length': 256, 'n_features': 3},
            'text': {'tokenizer': {'max_length': 128}}
        }
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    seq_len = config['time_series']['max_length']
    n_features = config['time_series']['n_features']
    text_len = config['text']['tokenizer']['max_length']
    
    # Generate time series data
    time_series_data = []
    text_data = []
    
    for i in range(num_samples):
        # Create diverse time series patterns
        t = np.linspace(0, 10, seq_len)
        
        if i % 4 == 0:  # Trend
            ts = np.column_stack([
                t + np.random.normal(0, 0.1, seq_len),
                np.sin(t) + np.random.normal(0, 0.1, seq_len),
                np.random.normal(0, 0.1, seq_len)
            ])[:, :n_features]
            text = "The data shows an upward trend with periodic variations."
            
        elif i % 4 == 1:  # Seasonal
            ts = np.column_stack([
                np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, seq_len),
                np.cos(2 * np.pi * t) + np.random.normal(0, 0.1, seq_len),
                np.random.normal(0, 0.1, seq_len)
            ])[:, :n_features]
            text = "Strong seasonal patterns are evident in this time series."
            
        elif i % 4 == 2:  # Volatile
            returns = np.random.normal(0, 0.02, seq_len)
            ts = np.column_stack([
                np.cumsum(returns),
                np.abs(returns) * 100,
                np.random.normal(0, 0.1, seq_len)
            ])[:, :n_features]
            text = "High volatility and random fluctuations characterize this data."
            
        else:  # Smooth
            ts = np.column_stack([
                np.exp(-t/5) + np.random.normal(0, 0.05, seq_len),
                t**0.5 + np.random.normal(0, 0.05, seq_len),
                np.random.normal(0, 0.05, seq_len)
            ])[:, :n_features]
            text = "The series exhibits smooth exponential decay patterns."
        
        time_series_data.append(ts)
        text_data.append(text)
    
    return time_series_data, text_data


# Pytest markers
pytest.mark.slow = pytest.mark.slow
pytest.mark.gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
pytest.mark.integration = pytest.mark.integration


# Configure pytest
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add slow marker to tests that might be slow
        if "training" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
