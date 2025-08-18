#!/usr/bin/env python3
"""
Standalone Training Script for Multimodal LLM
Can be run directly from terminal with minimal setup
"""

import os
import sys
import json
import argparse
import logging
import torch
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
warnings.filterwarnings('ignore')

# Setup paths
def setup_project_paths():
    """Setup Python paths for imports."""
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir
    
    # Add src to path
    src_path = project_root / 'src'
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    sys.path.insert(0, str(project_root))
    
    return str(project_root)

# Early path setup
project_root = setup_project_paths()

def setup_logging(log_level: str = 'INFO', log_file: str = 'training.log'):
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / log_file, mode='a')
        ]
    )
    
    # Suppress some verbose loggers
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)

def load_config(config_dir: Path) -> Dict[str, Any]:
    """Load and merge configuration files."""
    import yaml
    
    config_files = [
        'data_config.yaml',
        'model_config.yaml',
        'pipeline_config.yaml', 
        'training_config.yaml'
    ]
    
    merged_config = {}
    
    for config_file in config_files:
        config_path = config_dir / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    merged_config.update(file_config)
    
    return merged_config

def check_environment() -> Dict[str, Any]:
    """Check system requirements and environment."""
    env_info = {
        'python_version': sys.version,
        'torch_version': torch.__version__ if 'torch' in sys.modules else 'Not available',
        'cuda_available': torch.cuda.is_available() if 'torch' in sys.modules else False,
        'platform': sys.platform,
        'timestamp': datetime.now().isoformat()
    }
    
    if env_info['cuda_available']:
        env_info['gpu_count'] = torch.cuda.device_count()
        env_info['gpu_name'] = torch.cuda.get_device_name(0)
        env_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Check required packages
    required_packages = ['torch', 'transformers', 'numpy', 'pandas', 'yaml', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    env_info['missing_packages'] = missing_packages
    env_info['environment_ready'] = len(missing_packages) == 0
    
    return env_info

def create_directories(base_dir: Path) -> Dict[str, Path]:
    """Create necessary directories for training."""
    directories = {
        'checkpoints': base_dir / 'checkpoints',
        'logs': base_dir / 'logs',
        'outputs': base_dir / 'outputs',
        'plots': base_dir / 'outputs' / 'plots'
    }
    
    for name, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
    
    return directories

def prepare_data_loaders(config: Dict[str, Any], domains: List[str]) -> tuple:
    """Prepare training and validation data loaders."""
    from data.data_loader import MultimodalDataModule
    
    # Update config with selected domains
    data_config = config.copy()
    data_config['domains'] = domains
    data_config['data_dir'] = str(Path(project_root) / 'data' / 'time_mmd')
    
    # Create data module
    data_module = MultimodalDataModule(data_config)
    data_module.setup('fit')
    
    # Get data loaders
    train_loader = data_module.train_dataloader(distributed=False)
    val_loader = data_module.val_dataloader(distributed=False)
    
    return train_loader, val_loader

def create_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Create and initialize the multimodal model."""
    from models.multimodal_model import MultimodalLLM
    
    # Initialize model
    model = MultimodalLLM(config)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    return model

def setup_optimizer_and_scheduler(model: torch.nn.Module, 
                                 config: Dict[str, Any], 
                                 num_training_steps: int) -> tuple:
    """Setup optimizer and learning rate scheduler."""
    # Get optimizer parameters
    learning_rate = config.get('learning_rate', 5e-5)
    weight_decay = config.get('weight_decay', 0.01)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create scheduler
    from transformers import get_linear_schedule_with_warmup
    
    warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler

def train_epoch(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                device: torch.device,
                epoch: int) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    from tqdm import tqdm
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        
        outputs = model(
            time_series=batch.get('time_series'),
            ts_attention_mask=batch.get('ts_attention_mask'),
            text_input_ids=batch.get('text_input_ids'),
            text_attention_mask=batch.get('text_attention_mask'),
            labels=batch.get('labels')
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return {'train_loss': avg_loss, 'learning_rate': scheduler.get_last_lr()[0]}

def validate_epoch(model: torch.nn.Module,
                  val_loader: torch.utils.data.DataLoader,
                  device: torch.device) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                time_series=batch.get('time_series'),
                ts_attention_mask=batch.get('ts_attention_mask'),
                text_input_ids=batch.get('text_input_ids'),
                text_attention_mask=batch.get('text_attention_mask'),
                labels=batch.get('labels')
            )
            
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / num_batches
    return {'val_loss': avg_loss}

def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler,
                   epoch: int,
                   metrics: Dict[str, float],
                   save_path: Path,
                   is_best: bool = False) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save checkpoint
    checkpoint_path = save_path / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = save_path / 'best_model.pt'
        torch.save(checkpoint, best_path)
        
        # Also save just the model
        model_path = save_path / 'model.pt'
        torch.save(model.state_dict(), model_path)

def train_multimodal_llm(args) -> bool:
    """Main training function."""
    logger = logging.getLogger(__name__)
    
    print("🚀 STANDALONE MULTIMODAL LLM TRAINING")
    print("=" * 60)
    
    try:
        # Setup directories
        project_path = Path(project_root)
        directories = create_directories(project_path)
        
        # Load configuration
        config = load_config(project_path / 'config')
        
        # Override with command line arguments
        config.update({
            'domains': args.domains,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'save_every_n_epochs': args.save_every
        })
        
        logger.info(f"Training configuration: {config}")
        
        # Check environment
        env_info = check_environment()
        logger.info(f"Environment info: {env_info}")
        
        if not env_info['environment_ready']:
            print(f"❌ Missing packages: {env_info['missing_packages']}")
            return False
        
        print(f"✅ Environment ready")
        if env_info['cuda_available']:
            print(f"🚀 GPU: {env_info['gpu_name']} ({env_info['gpu_memory']:.1f} GB)")
        
        # Prepare data
        print(f"📊 Loading data for domains: {args.domains}")
        train_loader, val_loader = prepare_data_loaders(config, args.domains)
        
        print(f"📈 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
        print(f"🔄 Training batches: {len(train_loader)}")
        print(f"🔄 Validation batches: {len(val_loader)}")
        
        # Create model
        print("🤖 Creating model...")
        model = create_model(config)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Device: {device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"🔢 Total parameters: {total_params:,}")
        print(f"🔢 Trainable parameters: {trainable_params:,}")
        
        # Setup optimizer and scheduler
        num_training_steps = len(train_loader) * args.epochs
        optimizer, scheduler = setup_optimizer_and_scheduler(model, config, num_training_steps)
        
        print(f"⚙️ Optimizer: AdamW (lr={args.learning_rate})")
        print(f"📅 Training steps: {num_training_steps}")
        
        # Training loop
        print(f"\n🏋️ Starting training for {args.epochs} epochs...")
        
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(1, args.epochs + 1):
            print(f"\n--- Epoch {epoch}/{args.epochs} ---")
            
            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
            
            # Validate
            val_metrics = validate_epoch(model, val_loader, device)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            training_history.append(epoch_metrics)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Learning Rate: {train_metrics['learning_rate']:.2e}")
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
                print("🌟 New best model!")
            
            if epoch % args.save_every == 0 or is_best:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, 
                    epoch_metrics, directories['checkpoints'], is_best
                )
                print(f"💾 Checkpoint saved")
        
        # Save training history
        history_path = directories['outputs'] / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save final summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'domains': args.domains,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'best_val_loss': best_val_loss,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'training_time_epochs': args.epochs,
            'final_metrics': training_history[-1] if training_history else {},
            'environment_info': env_info,
            'model_path': str(directories['checkpoints'] / 'best_model.pt')
        }
        
        summary_path = directories['outputs'] / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n🎉 Training completed!")
        print(f"🏆 Best validation loss: {best_val_loss:.4f}")
        print(f"💾 Model saved to: {directories['checkpoints'] / 'best_model.pt'}")
        print(f"📊 Summary saved to: {summary_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"❌ Training failed: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Standalone Multimodal LLM Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training
  python train_standalone.py --domains Agriculture --epochs 2 --batch-size 1

  # Full training  
  python train_standalone.py --domains Agriculture Climate Economy --epochs 5 --batch-size 4

  # High learning rate training
  python train_standalone.py --domains Climate --epochs 3 --learning-rate 1e-4
        """
    )
    
    parser.add_argument('--domains', nargs='+', default=['Agriculture'],
                       help='Domains to train on')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--save-every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, 'standalone_training.log')
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting standalone training with args: {args}")
    
    # Run training
    try:
        success = train_multimodal_llm(args)
        
        if success:
            print("\n✅ Standalone training completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Standalone training failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()