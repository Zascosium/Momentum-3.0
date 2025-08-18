#!/usr/bin/env python3
"""
Standalone Data Loading and Preprocessing Check
Quick script to test data loading and preprocessing without running the full pipeline
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_paths():
    """Setup Python paths for imports."""
    project_root = Path(__file__).parent
    src_path = project_root / 'src'
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

def check_data_structure(data_dir: str) -> dict:
    """Check the structure of the data directory."""
    logger.info(f"🔍 Checking data structure in: {data_dir}")
    
    results = {
        'data_dir_exists': os.path.exists(data_dir),
        'domains': [],
        'files_found': {}
    }
    
    if not results['data_dir_exists']:
        logger.error(f"❌ Data directory not found: {data_dir}")
        return results
    
    # Check for domains
    numerical_dir = os.path.join(data_dir, 'numerical')
    textual_dir = os.path.join(data_dir, 'textual')
    
    if os.path.exists(numerical_dir):
        results['domains'] = [d for d in os.listdir(numerical_dir) if os.path.isdir(os.path.join(numerical_dir, d))]
        logger.info(f"✅ Found domains: {results['domains']}")
    
    # Check files in each domain
    for domain in results['domains']:
        domain_files = {}
        
        # Check numerical data
        num_domain_dir = os.path.join(numerical_dir, domain)
        if os.path.exists(num_domain_dir):
            domain_files['numerical'] = os.listdir(num_domain_dir)
        
        # Check textual data
        text_domain_dir = os.path.join(textual_dir, domain)
        if os.path.exists(text_domain_dir):
            domain_files['textual'] = os.listdir(text_domain_dir)
        
        results['files_found'][domain] = domain_files
        logger.info(f"📁 {domain}: {len(domain_files.get('numerical', []))} numerical files, {len(domain_files.get('textual', []))} textual files")
    
    return results

def test_data_loading(domains: List[str] = None, sample_size: int = 5):
    """Test data loading functionality."""
    logger.info("🔄 Testing data loading...")
    
    try:
        # Import data loading modules
        from data.data_loader import MultimodalDataModule
        from utils.config_loader import load_config_for_training
        
        # Load configuration
        config = load_config_for_training()
        
        # Override domains if specified
        if domains:
            config['data']['domains'] = domains
        
        # Create data module
        data_module = MultimodalDataModule(config)
        data_module.setup('fit')
        
        # Test loading a few samples
        train_loader = data_module.train_dataloader(distributed=False)
        
        logger.info(f"✅ Data module created successfully")
        logger.info(f"📊 Training batches: {len(train_loader)}")
        
        # Load and inspect a few samples
        logger.info(f"🔍 Loading {sample_size} sample batches...")
        
        for i, batch in enumerate(train_loader):
            if i >= sample_size:
                break
                
            logger.info(f"Batch {i+1}:")
            for key, value in batch.items():
                if hasattr(value, 'shape'):
                    logger.info(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                elif isinstance(value, (list, tuple)):
                    logger.info(f"  {key}: {len(value)} items")
                else:
                    logger.info(f"  {key}: {type(value)}")
        
        logger.info("✅ Data loading test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        logger.error("This might be due to missing data files or configuration issues")
        return False

def test_preprocessing(domains: List[str] = None):
    """Test preprocessing functionality."""
    logger.info("🔄 Testing preprocessing...")
    
    try:
        # Import preprocessing modules
        from data.preprocessing import TimeSeriesPreprocessor, TextPreprocessor
        
        # Test time series preprocessor
        ts_preprocessor = TimeSeriesPreprocessor()
        logger.info("✅ TimeSeriesPreprocessor created")
        
        # Test text preprocessor
        text_preprocessor = TextPreprocessor()
        logger.info("✅ TextPreprocessor created")
        
        logger.info("✅ Preprocessing test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Preprocessing test failed: {e}")
        return False

def main():
    """Main function to run data checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check data loading and preprocessing")
    parser.add_argument('--domains', nargs='+', default=['Agriculture', 'Climate'], 
                       help='Domains to check')
    parser.add_argument('--data-dir', type=str, 
                       default='data/time_mmd', help='Data directory path')
    parser.add_argument('--sample-size', type=int, default=3,
                       help='Number of sample batches to load')
    parser.add_argument('--skip-loading', action='store_true',
                       help='Skip data loading test')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                       help='Skip preprocessing test')
    
    args = parser.parse_args()
    
    # Setup paths
    setup_paths()
    
    logger.info("🚀 Starting Data Loading and Preprocessing Check")
    logger.info("=" * 60)
    
    # Check data structure
    data_results = check_data_structure(args.data_dir)
    
    if not data_results['data_dir_exists']:
        logger.error("Cannot proceed without data directory")
        return 1
    
    if not data_results['domains']:
        logger.warning("No domains found in data directory")
        return 1
    
    # Test preprocessing
    if not args.skip_preprocessing:
        preprocessing_ok = test_preprocessing(args.domains)
        if not preprocessing_ok:
            logger.error("Preprocessing test failed")
            return 1
    
    # Test data loading
    if not args.skip_loading:
        loading_ok = test_data_loading(args.domains, args.sample_size)
        if not loading_ok:
            logger.error("Data loading test failed")
            return 1
    
    logger.info("🎉 All data checks completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
