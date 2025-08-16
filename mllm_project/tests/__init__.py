"""
Test suite for the Multimodal LLM project.

This package contains comprehensive tests for all components of the MLLM system,
including unit tests, integration tests, and end-to-end tests.

Test Structure:
- test_models/: Tests for model components
- test_data/: Tests for data processing and loading
- test_training/: Tests for training infrastructure
- test_utils/: Tests for utility functions
- test_integration/: Integration and end-to-end tests

Usage:
    # Run all tests
    python -m pytest tests/

    # Run specific test module
    python -m pytest tests/test_models/

    # Run with coverage
    python -m pytest tests/ --cov=src --cov-report=html
"""

__version__ = "1.0.0"
__author__ = "MLLM Team"
