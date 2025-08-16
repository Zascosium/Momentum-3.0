"""
Tests for data preprocessing modules.

This module contains tests for time series and text preprocessing functionality.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch
from typing import List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data.preprocessing import (
    TimeSeriesPreprocessor, 
    TextPreprocessor,
    normalize_time_series,
    handle_missing_values,
    detect_outliers,
    align_time_series_text
)


class TestTimeSeriesPreprocessor:
    """Test suite for TimeSeriesPreprocessor."""
    
    def test_initialization(self, test_config):
        """Test preprocessor initialization."""
        ts_config = test_config['time_series']
        preprocessor = TimeSeriesPreprocessor(ts_config)
        
        assert preprocessor.max_length == ts_config['max_length']
        assert preprocessor.n_features == ts_config['n_features']
        assert preprocessor.normalization == ts_config['normalization']
    
    def test_normalize_standard(self, sample_time_series):
        """Test standard normalization."""
        ts_config = {
            'normalization': 'standard',
            'max_length': 256,
            'n_features': 3
        }
        preprocessor = TimeSeriesPreprocessor(ts_config)
        
        normalized = preprocessor.normalize(sample_time_series)
        
        # Check that mean is close to 0 and std is close to 1
        assert np.abs(normalized.mean(axis=0)).max() < 0.1
        assert np.abs(normalized.std(axis=0) - 1.0).max() < 0.1
    
    def test_normalize_minmax(self, sample_time_series):
        """Test min-max normalization."""
        ts_config = {
            'normalization': 'minmax',
            'max_length': 256,
            'n_features': 3
        }
        preprocessor = TimeSeriesPreprocessor(ts_config)
        
        normalized = preprocessor.normalize(sample_time_series)
        
        # Check that values are between 0 and 1
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_pad_or_truncate_pad(self):
        """Test padding short sequences."""
        ts_config = {
            'max_length': 100,
            'n_features': 2,
            'normalization': 'none'
        }
        preprocessor = TimeSeriesPreprocessor(ts_config)
        
        # Short sequence
        short_ts = np.random.randn(50, 2)
        padded, mask = preprocessor.pad_or_truncate(short_ts)
        
        assert padded.shape == (100, 2)
        assert mask.shape == (100,)
        assert mask[:50].all()  # First 50 should be True
        assert not mask[50:].any()  # Last 50 should be False
    
    def test_pad_or_truncate_truncate(self):
        """Test truncating long sequences."""
        ts_config = {
            'max_length': 100,
            'n_features': 2,
            'normalization': 'none'
        }
        preprocessor = TimeSeriesPreprocessor(ts_config)
        
        # Long sequence
        long_ts = np.random.randn(150, 2)
        truncated, mask = preprocessor.pad_or_truncate(long_ts)
        
        assert truncated.shape == (100, 2)
        assert mask.shape == (100,)
        assert mask.all()  # All should be True
    
    def test_handle_missing_values_interpolate(self):
        """Test missing value handling with interpolation."""
        ts_config = {
            'preprocessing': {'fill_missing': 'interpolate'},
            'max_length': 256,
            'n_features': 3,
            'normalization': 'none'
        }
        preprocessor = TimeSeriesPreprocessor(ts_config)
        
        # Create data with missing values
        data = np.random.randn(100, 3)
        data[20:25, 1] = np.nan  # Insert missing values
        data[50, :] = np.nan
        
        filled = preprocessor.handle_missing_values(data)
        
        # Check no NaN values remain
        assert not np.isnan(filled).any()
        assert filled.shape == data.shape
    
    def test_handle_missing_values_forward_fill(self):
        """Test missing value handling with forward fill."""
        ts_config = {
            'preprocessing': {'fill_missing': 'forward'},
            'max_length': 256,
            'n_features': 3,
            'normalization': 'none'
        }
        preprocessor = TimeSeriesPreprocessor(ts_config)
        
        # Create data with missing values
        data = np.array([[1.0, 2.0], [3.0, 4.0], [np.nan, np.nan], [7.0, 8.0]])
        filled = preprocessor.handle_missing_values(data)
        
        # Check forward fill worked
        assert not np.isnan(filled).any()
        assert filled[2, 0] == 3.0  # Forward filled
        assert filled[2, 1] == 4.0  # Forward filled
    
    def test_detect_outliers(self, sample_time_series):
        """Test outlier detection."""
        ts_config = {
            'preprocessing': {
                'outlier_detection': True,
                'outlier_threshold': 3.0
            },
            'max_length': 256,
            'n_features': 3,
            'normalization': 'none'
        }
        preprocessor = TimeSeriesPreprocessor(ts_config)
        
        # Add some outliers
        data_with_outliers = sample_time_series.copy()
        data_with_outliers[10, 0] = 100.0  # Clear outlier
        data_with_outliers[50, 1] = -100.0  # Clear outlier
        
        outliers = preprocessor.detect_outliers(data_with_outliers)
        
        assert isinstance(outliers, np.ndarray)
        assert outliers.dtype == bool
        assert outliers.shape == data_with_outliers.shape
        assert outliers[10, 0]  # Should detect outlier
        assert outliers[50, 1]  # Should detect outlier
    
    def test_preprocess_complete(self, sample_time_series):
        """Test complete preprocessing pipeline."""
        ts_config = {
            'max_length': 128,
            'n_features': 3,
            'normalization': 'standard',
            'preprocessing': {
                'fill_missing': 'interpolate',
                'outlier_detection': True,
                'outlier_threshold': 3.0
            }
        }
        preprocessor = TimeSeriesPreprocessor(ts_config)
        
        # Add some missing values and outliers
        data = sample_time_series.copy()
        data[10:15, 1] = np.nan
        data[50, 0] = 100.0
        
        processed, mask = preprocessor.preprocess(data)
        
        assert processed.shape == (128, 3)
        assert mask.shape == (128,)
        assert not np.isnan(processed).any()
        assert isinstance(processed, np.ndarray)
        assert isinstance(mask, np.ndarray)
    
    def test_batch_preprocess(self, sample_time_series):
        """Test batch preprocessing."""
        ts_config = {
            'max_length': 100,
            'n_features': 3,
            'normalization': 'standard'
        }
        preprocessor = TimeSeriesPreprocessor(ts_config)
        
        # Create batch of time series
        batch = [sample_time_series[:80], sample_time_series[:120], sample_time_series[:60]]
        
        processed_batch, masks = preprocessor.preprocess_batch(batch)
        
        assert processed_batch.shape == (3, 100, 3)
        assert masks.shape == (3, 100)
        assert isinstance(processed_batch, np.ndarray)
        assert isinstance(masks, np.ndarray)


class TestTextPreprocessor:
    """Test suite for TextPreprocessor."""
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_initialization(self, mock_tokenizer, test_config):
        """Test text preprocessor initialization."""
        # Setup mock tokenizer
        mock_instance = Mock()
        mock_instance.vocab_size = 50257
        mock_instance.pad_token_id = 0
        mock_instance.eos_token_id = 1
        mock_tokenizer.return_value = mock_instance
        
        text_config = test_config['text']
        preprocessor = TextPreprocessor(text_config)
        
        assert preprocessor.max_length == text_config['tokenizer']['max_length']
        assert hasattr(preprocessor, 'tokenizer')
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_clean_text(self, mock_tokenizer, test_config):
        """Test text cleaning functionality."""
        mock_instance = Mock()
        mock_tokenizer.return_value = mock_instance
        
        text_config = test_config['text']
        text_config['preprocessing']['lowercase'] = True
        text_config['preprocessing']['remove_special_chars'] = True
        
        preprocessor = TextPreprocessor(text_config)
        
        dirty_text = "Hello WORLD!!! This is a TEST @#$%"
        cleaned = preprocessor.clean_text(dirty_text)
        
        assert isinstance(cleaned, str)
        assert cleaned.islower()
        # Check that some cleaning occurred
        assert len(cleaned) <= len(dirty_text)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenize_single(self, mock_tokenizer, test_config):
        """Test single text tokenization."""
        # Setup mock tokenizer
        mock_instance = Mock()
        mock_instance.encode.return_value = [1, 2, 3, 4, 5]
        mock_instance.vocab_size = 50257
        mock_tokenizer.return_value = mock_instance
        
        preprocessor = TextPreprocessor(test_config['text'])
        
        text = "This is a test sentence."
        tokens = preprocessor.tokenize(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        mock_instance.encode.assert_called_once()
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenize_batch(self, mock_tokenizer, test_config, sample_text_data):
        """Test batch text tokenization."""
        # Setup mock tokenizer
        mock_instance = Mock()
        mock_instance.batch_encode_plus.return_value = {
            'input_ids': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            'attention_mask': [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        }
        mock_instance.vocab_size = 50257
        mock_tokenizer.return_value = mock_instance
        
        preprocessor = TextPreprocessor(test_config['text'])
        
        result = preprocessor.tokenize_batch(sample_text_data[:3])
        
        assert isinstance(result, dict)
        assert 'input_ids' in result
        assert 'attention_mask' in result
        mock_instance.batch_encode_plus.assert_called_once()
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_decode(self, mock_tokenizer, test_config):
        """Test token decoding."""
        # Setup mock tokenizer
        mock_instance = Mock()
        mock_instance.decode.return_value = "This is decoded text"
        mock_tokenizer.return_value = mock_instance
        
        preprocessor = TextPreprocessor(test_config['text'])
        
        tokens = [1, 2, 3, 4, 5]
        decoded = preprocessor.decode(tokens)
        
        assert isinstance(decoded, str)
        assert len(decoded) > 0
        mock_instance.decode.assert_called_once_with(tokens, skip_special_tokens=True)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_preprocess_complete(self, mock_tokenizer, test_config):
        """Test complete text preprocessing pipeline."""
        # Setup mock tokenizer
        mock_instance = Mock()
        mock_instance.encode.return_value = [1, 2, 3, 4, 5, 0, 0, 0]  # Padded
        mock_instance.vocab_size = 50257
        mock_instance.pad_token_id = 0
        mock_tokenizer.return_value = mock_instance
        
        preprocessor = TextPreprocessor(test_config['text'])
        
        text = "This is a TEST sentence with CAPS!"
        processed = preprocessor.preprocess(text)
        
        assert isinstance(processed, dict)
        assert 'input_ids' in processed
        assert 'attention_mask' in processed
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_filter_by_length(self, mock_tokenizer, test_config):
        """Test text filtering by length."""
        mock_instance = Mock()
        mock_tokenizer.return_value = mock_instance
        
        text_config = test_config['text']
        text_config['preprocessing']['min_length'] = 10
        text_config['preprocessing']['max_length'] = 100
        
        preprocessor = TextPreprocessor(text_config)
        
        texts = [
            "Short",  # Too short
            "This is a medium length sentence that should pass the filter.",  # Good
            "A" * 150,  # Too long
            "Another good sentence for testing."  # Good
        ]
        
        filtered = preprocessor.filter_by_length(texts)
        
        assert len(filtered) == 2  # Should keep 2 texts
        assert "Short" not in filtered
        assert "A" * 150 not in filtered


class TestUtilityFunctions:
    """Test utility functions for preprocessing."""
    
    def test_normalize_time_series_standard(self, sample_time_series):
        """Test standard normalization utility."""
        normalized = normalize_time_series(sample_time_series, method='standard')
        
        assert normalized.shape == sample_time_series.shape
        assert np.abs(normalized.mean(axis=0)).max() < 0.1
        assert np.abs(normalized.std(axis=0) - 1.0).max() < 0.1
    
    def test_normalize_time_series_minmax(self, sample_time_series):
        """Test min-max normalization utility."""
        normalized = normalize_time_series(sample_time_series, method='minmax')
        
        assert normalized.shape == sample_time_series.shape
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
    
    def test_handle_missing_values_interpolate(self):
        """Test missing value handling utility."""
        # Create data with missing values
        data = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0], [np.nan, 8.0]])
        
        filled = handle_missing_values(data, method='interpolate')
        
        assert not np.isnan(filled).any()
        assert filled.shape == data.shape
        # Check interpolation worked
        assert filled[1, 1] == 4.0  # Should be interpolated
    
    def test_detect_outliers_zscore(self, sample_time_series):
        """Test outlier detection utility."""
        # Add clear outliers
        data_with_outliers = sample_time_series.copy()
        data_with_outliers[10, 0] = 100.0
        data_with_outliers[50, 1] = -100.0
        
        outliers = detect_outliers(data_with_outliers, method='zscore', threshold=3.0)
        
        assert isinstance(outliers, np.ndarray)
        assert outliers.dtype == bool
        assert outliers.shape == data_with_outliers.shape
        assert outliers[10, 0]  # Should detect outlier
        assert outliers[50, 1]  # Should detect outlier
    
    def test_align_time_series_text(self):
        """Test time series and text alignment utility."""
        # Create sample data with timestamps
        ts_data = [
            {'timestamp': '2023-01-01 10:00:00', 'values': [1, 2, 3]},
            {'timestamp': '2023-01-01 11:00:00', 'values': [4, 5, 6]},
            {'timestamp': '2023-01-01 12:00:00', 'values': [7, 8, 9]}
        ]
        
        text_data = [
            {'timestamp': '2023-01-01 10:05:00', 'text': 'Morning report'},
            {'timestamp': '2023-01-01 11:02:00', 'text': 'Midday update'},
            {'timestamp': '2023-01-01 12:30:00', 'text': 'Afternoon summary'}
        ]
        
        aligned = align_time_series_text(ts_data, text_data, tolerance_minutes=30)
        
        assert isinstance(aligned, list)
        assert len(aligned) > 0
        
        for pair in aligned:
            assert 'time_series' in pair
            assert 'text' in pair
            assert 'ts_timestamp' in pair
            assert 'text_timestamp' in pair


@pytest.mark.integration
class TestPreprocessingIntegration:
    """Integration tests for preprocessing components."""
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_complete_preprocessing_pipeline(self, mock_tokenizer, test_config, 
                                           sample_time_series, sample_text_data):
        """Test complete preprocessing pipeline integration."""
        # Setup mock tokenizer
        mock_instance = Mock()
        mock_instance.batch_encode_plus.return_value = {
            'input_ids': [[1, 2, 3, 0, 0]] * len(sample_text_data),
            'attention_mask': [[1, 1, 1, 0, 0]] * len(sample_text_data)
        }
        mock_instance.vocab_size = 50257
        mock_tokenizer.return_value = mock_instance
        
        # Initialize preprocessors
        ts_preprocessor = TimeSeriesPreprocessor(test_config['time_series'])
        text_preprocessor = TextPreprocessor(test_config['text'])
        
        # Process time series batch
        ts_batch = [sample_time_series[:100], sample_time_series[:150], sample_time_series[:80]]
        processed_ts, ts_masks = ts_preprocessor.preprocess_batch(ts_batch)
        
        # Process text batch
        processed_text = text_preprocessor.tokenize_batch(sample_text_data[:3])
        
        # Verify shapes and types
        assert processed_ts.shape[0] == 3  # Batch size
        assert processed_ts.shape[1] == test_config['time_series']['max_length']
        assert processed_ts.shape[2] == test_config['time_series']['n_features']
        
        assert len(processed_text['input_ids']) == 3
        assert len(processed_text['attention_mask']) == 3
    
    def test_preprocessing_with_real_data_simulation(self, test_config):
        """Test preprocessing with simulated real-world data."""
        # Simulate realistic time series data
        np.random.seed(42)
        
        # Stock price-like data
        returns = np.random.normal(0.001, 0.02, 500)
        prices = 100 * np.exp(np.cumsum(returns))
        volumes = np.random.lognormal(10, 0.5, 500)
        volatility = np.abs(returns) * 100
        
        real_ts = np.column_stack([prices, volumes, volatility])
        
        # Add missing values and outliers
        real_ts[100:105, 1] = np.nan  # Missing volume data
        real_ts[200, 0] = real_ts[200, 0] * 5  # Price spike
        
        # Process with preprocessor
        ts_preprocessor = TimeSeriesPreprocessor(test_config['time_series'])
        processed, mask = ts_preprocessor.preprocess(real_ts)
        
        assert not np.isnan(processed).any()
        assert processed.shape[1] == test_config['time_series']['n_features']
        assert mask.sum() > 0  # Some valid timesteps
