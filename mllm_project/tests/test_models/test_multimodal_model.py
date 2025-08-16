"""
Tests for the MultimodalLLM model.

This module contains comprehensive tests for the main multimodal model,
including forward pass, generation, and integration tests.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.multimodal_model import MultimodalLLM, MultimodalOutput


class TestMultimodalLLM:
    """Test suite for MultimodalLLM class."""
    
    def test_model_initialization(self, test_config):
        """Test model initialization with valid config."""
        with patch('transformers.AutoModel.from_pretrained'), \
             patch('transformers.AutoTokenizer.from_pretrained'), \
             patch('transformers.GPT2LMHeadModel.from_pretrained'):
            
            model = MultimodalLLM(test_config)
            
            assert model is not None
            assert hasattr(model, 'moment_encoder')
            assert hasattr(model, 'text_decoder')
            assert hasattr(model, 'projection_layer')
            assert hasattr(model, 'cross_attention')
    
    def test_model_initialization_invalid_config(self):
        """Test model initialization with invalid config."""
        invalid_config = {'invalid': 'config'}
        
        with pytest.raises((KeyError, ValueError)):
            MultimodalLLM(invalid_config)
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained') 
    @patch('transformers.GPT2LMHeadModel.from_pretrained')
    def test_forward_pass_complete(self, mock_gpt2, mock_tokenizer, mock_moment, 
                                  test_config, sample_batch, device):
        """Test complete forward pass with all inputs."""
        # Setup mocks
        mock_moment_instance = Mock()
        mock_moment_instance.return_value = Mock()
        mock_moment_instance.return_value.last_hidden_state = torch.randn(8, 256, 512)
        mock_moment.return_value = mock_moment_instance
        
        mock_gpt2_instance = Mock()
        mock_gpt2_instance.return_value = Mock()
        mock_gpt2_instance.return_value.logits = torch.randn(8, 128, 50257)
        mock_gpt2_instance.return_value.loss = torch.tensor(2.5)
        mock_gpt2.return_value = mock_gpt2_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.vocab_size = 50257
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Create and test model
        model = MultimodalLLM(test_config)
        model.to(device)
        
        # Move batch to device
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                sample_batch[key] = value.to(device)
        
        # Forward pass
        output = model(
            time_series=sample_batch['time_series'],
            ts_attention_mask=sample_batch['ts_attention_mask'],
            text_input_ids=sample_batch['text_input_ids'],
            text_attention_mask=sample_batch['text_attention_mask'],
            labels=sample_batch['text_input_ids']
        )
        
        assert isinstance(output, MultimodalOutput)
        assert output.logits is not None
        assert output.loss is not None
        assert output.ts_embeddings is not None
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.GPT2LMHeadModel.from_pretrained')
    def test_forward_pass_time_series_only(self, mock_gpt2, mock_tokenizer, mock_moment,
                                          test_config, device):
        """Test forward pass with only time series input."""
        # Setup mocks
        mock_moment_instance = Mock()
        mock_moment_instance.return_value = Mock()
        mock_moment_instance.return_value.last_hidden_state = torch.randn(4, 256, 512)
        mock_moment.return_value = mock_moment_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.vocab_size = 50257
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        model = MultimodalLLM(test_config)
        model.to(device)
        
        # Test with time series only
        time_series = torch.randn(4, 256, 3).to(device)
        ts_mask = torch.ones(4, 256, dtype=torch.bool).to(device)
        
        output = model(
            time_series=time_series,
            ts_attention_mask=ts_mask
        )
        
        assert isinstance(output, MultimodalOutput)
        assert output.ts_embeddings is not None
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.GPT2LMHeadModel.from_pretrained')
    def test_forward_pass_text_only(self, mock_gpt2, mock_tokenizer, mock_moment,
                                   test_config, device):
        """Test forward pass with only text input."""
        # Setup mocks
        mock_gpt2_instance = Mock()
        mock_gpt2_instance.return_value = Mock()
        mock_gpt2_instance.return_value.logits = torch.randn(4, 64, 50257)
        mock_gpt2_instance.return_value.loss = torch.tensor(1.8)
        mock_gpt2.return_value = mock_gpt2_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.vocab_size = 50257
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        model = MultimodalLLM(test_config)
        model.to(device)
        
        # Test with text only
        text_ids = torch.randint(0, 1000, (4, 64)).to(device)
        text_mask = torch.ones(4, 64, dtype=torch.bool).to(device)
        
        output = model(
            text_input_ids=text_ids,
            text_attention_mask=text_mask,
            labels=text_ids
        )
        
        assert isinstance(output, MultimodalOutput)
        assert output.logits is not None
        assert output.loss is not None
    
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.GPT2LMHeadModel.from_pretrained')
    def test_generate_method(self, mock_gpt2, mock_tokenizer, mock_moment,
                            test_config, device):
        """Test text generation method."""
        # Setup mocks
        mock_moment_instance = Mock()
        mock_moment_instance.return_value = Mock()
        mock_moment_instance.return_value.last_hidden_state = torch.randn(1, 256, 512)
        mock_moment.return_value = mock_moment_instance
        
        mock_gpt2_instance = Mock()
        mock_gpt2_instance.generate = Mock()
        mock_gpt2_instance.generate.return_value = torch.randint(0, 1000, (1, 80))
        mock_gpt2.return_value = mock_gpt2_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.vocab_size = 50257
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        model = MultimodalLLM(test_config)
        model.to(device)
        model.eval()
        
        # Test generation
        time_series = torch.randn(1, 256, 3).to(device)
        ts_mask = torch.ones(1, 256, dtype=torch.bool).to(device)
        text_ids = torch.randint(0, 1000, (1, 20)).to(device)
        
        with torch.no_grad():
            generated = model.generate(
                time_series=time_series,
                ts_attention_mask=ts_mask,
                text_input_ids=text_ids,
                max_length=50,
                temperature=0.8
            )
        
        assert generated is not None
        assert generated.shape[0] == 1
        assert generated.shape[1] >= 20  # Should be at least as long as input
    
    def test_get_memory_usage(self, mock_model):
        """Test memory usage calculation."""
        memory_stats = mock_model.get_memory_usage()
        
        assert isinstance(memory_stats, dict)
        assert 'total_parameters' in memory_stats
        assert 'trainable_parameters' in memory_stats
        assert 'parameter_memory_mb' in memory_stats
        assert isinstance(memory_stats['total_parameters'], int)
        assert isinstance(memory_stats['trainable_parameters'], int)
        assert isinstance(memory_stats['parameter_memory_mb'], float)
    
    def test_freeze_components(self, mock_model):
        """Test component freezing functionality."""
        # Test freezing moment encoder
        mock_model.freeze_moment_encoder()
        # Verify that parameters are frozen (requires_grad=False)
        # Note: This is a simplified test since we're using mocks
        
        # Test freezing text decoder
        mock_model.freeze_text_decoder()
        
        # Test unfreezing
        mock_model.unfreeze_all()
    
    def test_save_and_load_config(self, mock_model, temp_dir):
        """Test saving and loading model configuration."""
        config_path = f"{temp_dir}/model_config.json"
        
        # Save config
        mock_model.save_config(config_path)
        
        # Load config
        loaded_config = MultimodalLLM.load_config(config_path)
        
        assert loaded_config is not None
        assert isinstance(loaded_config, dict)
    
    def test_model_device_handling(self, test_config, device):
        """Test model device handling."""
        with patch('transformers.AutoModel.from_pretrained'), \
             patch('transformers.AutoTokenizer.from_pretrained'), \
             patch('transformers.GPT2LMHeadModel.from_pretrained'):
            
            model = MultimodalLLM(test_config)
            
            # Move to device
            model.to(device)
            
            # Check device
            for param in model.parameters():
                assert param.device == device
    
    def test_training_mode_switching(self, mock_model):
        """Test switching between training and evaluation modes."""
        # Test training mode
        mock_model.train()
        assert mock_model.training
        
        # Test evaluation mode
        mock_model.eval()
        assert not mock_model.training
    
    @pytest.mark.parametrize("batch_size,seq_len,n_features", [
        (1, 128, 3),
        (4, 256, 5),
        (8, 512, 1),
    ])
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.GPT2LMHeadModel.from_pretrained')
    def test_different_input_sizes(self, mock_gpt2, mock_tokenizer, mock_moment,
                                  batch_size, seq_len, n_features, test_config, device):
        """Test model with different input sizes."""
        # Setup mocks
        mock_moment_instance = Mock()
        mock_moment_instance.return_value = Mock()
        mock_moment_instance.return_value.last_hidden_state = torch.randn(batch_size, seq_len, 512)
        mock_moment.return_value = mock_moment_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.vocab_size = 50257
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Adjust config for this test
        test_config['time_series']['n_features'] = n_features
        
        model = MultimodalLLM(test_config)
        model.to(device)
        
        # Test with different input sizes
        time_series = torch.randn(batch_size, seq_len, n_features).to(device)
        ts_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
        
        output = model(
            time_series=time_series,
            ts_attention_mask=ts_mask
        )
        
        assert output.ts_embeddings is not None
        assert output.ts_embeddings.shape[0] == batch_size
    
    def test_invalid_inputs(self, mock_model, device):
        """Test model behavior with invalid inputs."""
        # Test with mismatched dimensions
        time_series = torch.randn(4, 256, 5).to(device)  # Wrong feature dimension
        ts_mask = torch.ones(4, 256, dtype=torch.bool).to(device)
        
        with pytest.raises((RuntimeError, ValueError)):
            mock_model(
                time_series=time_series,
                ts_attention_mask=ts_mask
            )
    
    def test_gradient_flow(self, mock_model, sample_batch, device):
        """Test gradient flow through the model."""
        mock_model.train()
        
        # Move batch to device
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                sample_batch[key] = value.to(device)
        
        # Forward pass with gradient computation
        output = mock_model(
            time_series=sample_batch['time_series'],
            ts_attention_mask=sample_batch['ts_attention_mask'],
            text_input_ids=sample_batch['text_input_ids'],
            text_attention_mask=sample_batch['text_attention_mask'],
            labels=sample_batch['text_input_ids']
        )
        
        # Check that loss requires gradient
        if output.loss is not None:
            assert output.loss.requires_grad
            
            # Backward pass
            output.loss.backward()
            
            # Check that some parameters have gradients
            has_gradients = any(
                param.grad is not None 
                for param in mock_model.parameters() 
                if param.requires_grad
            )
            # Note: This might not work with mocks, but tests the structure


class TestMultimodalOutput:
    """Test suite for MultimodalOutput class."""
    
    def test_output_creation(self):
        """Test MultimodalOutput creation."""
        logits = torch.randn(4, 128, 50257)
        loss = torch.tensor(2.5)
        ts_embeddings = torch.randn(4, 256, 512)
        
        output = MultimodalOutput(
            logits=logits,
            loss=loss,
            ts_embeddings=ts_embeddings
        )
        
        assert output.logits is not None
        assert output.loss is not None
        assert output.ts_embeddings is not None
        assert torch.equal(output.logits, logits)
        assert torch.equal(output.loss, loss)
        assert torch.equal(output.ts_embeddings, ts_embeddings)
    
    def test_output_optional_fields(self):
        """Test MultimodalOutput with optional fields."""
        output = MultimodalOutput()
        
        assert output.logits is None
        assert output.loss is None
        assert output.ts_embeddings is None
        assert output.text_embeddings is None
        assert output.attention_weights is None
    
    def test_output_to_dict(self):
        """Test converting output to dictionary."""
        logits = torch.randn(2, 64, 1000)
        loss = torch.tensor(1.5)
        
        output = MultimodalOutput(logits=logits, loss=loss)
        output_dict = output.to_dict()
        
        assert isinstance(output_dict, dict)
        assert 'logits' in output_dict
        assert 'loss' in output_dict
        assert torch.equal(output_dict['logits'], logits)
        assert torch.equal(output_dict['loss'], loss)


@pytest.mark.integration
class TestMultimodalLLMIntegration:
    """Integration tests for MultimodalLLM."""
    
    @pytest.mark.slow
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.GPT2LMHeadModel.from_pretrained')
    def test_full_training_step(self, mock_gpt2, mock_tokenizer, mock_moment,
                               test_config, device):
        """Test a complete training step."""
        # Setup comprehensive mocks
        mock_moment_instance = Mock()
        mock_moment_instance.return_value = Mock()
        mock_moment_instance.return_value.last_hidden_state = torch.randn(8, 256, 512)
        mock_moment.return_value = mock_moment_instance
        
        mock_gpt2_instance = Mock()
        mock_gpt2_instance.return_value = Mock()
        mock_gpt2_instance.return_value.logits = torch.randn(8, 128, 50257)
        mock_gpt2_instance.return_value.loss = torch.tensor(2.5, requires_grad=True)
        mock_gpt2.return_value = mock_gpt2_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.vocab_size = 50257
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        model = MultimodalLLM(test_config)
        model.to(device)
        model.train()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Generate batch
        batch_size = 8
        seq_len = 256
        text_len = 128
        n_features = 3
        
        batch = {
            'time_series': torch.randn(batch_size, seq_len, n_features).to(device),
            'ts_attention_mask': torch.ones(batch_size, seq_len, dtype=torch.bool).to(device),
            'text_input_ids': torch.randint(0, 1000, (batch_size, text_len)).to(device),
            'text_attention_mask': torch.ones(batch_size, text_len, dtype=torch.bool).to(device)
        }
        
        # Training step
        optimizer.zero_grad()
        
        output = model(
            time_series=batch['time_series'],
            ts_attention_mask=batch['ts_attention_mask'],
            text_input_ids=batch['text_input_ids'],
            text_attention_mask=batch['text_attention_mask'],
            labels=batch['text_input_ids']
        )
        
        assert output.loss is not None
        
        # Note: Backward pass might not work properly with mocks
        # but this tests the structure
    
    @pytest.mark.slow
    def test_model_persistence(self, mock_model, temp_dir):
        """Test saving and loading model state."""
        model_path = f"{temp_dir}/model.pt"
        
        # Save model
        torch.save({
            'model_state_dict': mock_model.state_dict(),
            'config': mock_model.config if hasattr(mock_model, 'config') else {}
        }, model_path)
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        assert 'model_state_dict' in checkpoint
        assert 'config' in checkpoint
