# 🚀 Quick Start: Single Command Training

Run your entire multimodal LLM production pipeline with just one command!

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

## ⚡ Quick Start

### Option 1: Bash Script (Recommended)

```bash
# Quick training with defaults (3 epochs)
./run_training.sh

# Custom training
./run_training.sh --epochs 10 --batch-size 8 --experiment-name "My_Training"

# Show all options
./run_training.sh --help
```

### Option 2: Python Script

```bash
# Basic training
python run_production_training.py

# Custom training
python run_production_training.py --epochs 10 --batch-size 8
```

## 🎯 What It Does

The single command automatically:

1. **Environment Setup** ✅
   - Detects Databricks vs local environment
   - Validates Python and CUDA
   - Sets up directory structure

2. **Configuration** ⚙️
   - Creates production-ready config
   - Validates and fixes configuration issues
   - Optimizes for your hardware

3. **Data Preparation** 📊
   - Generates mock data for testing
   - Sets up data loaders
   - Handles preprocessing

4. **Model Training** 🏋️
   - Initializes multimodal model
   - Runs training with proper error handling
   - Logs metrics and progress

5. **Model Saving** 💾
   - Saves trained model and metadata
   - Logs to MLflow (if available)
   - Creates inference package

## 📊 Output

After completion, you'll find:

```
production_data/
├── models/final_model/          # Trained model files
├── config/                      # Configuration files  
├── logs/                        # Training logs
├── checkpoints/                 # Model checkpoints
└── data/                        # Training data

production_training.log          # Detailed execution log
```

## ⚙️ Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs N` | Number of training epochs | 3 |
| `--batch-size N` | Training batch size | auto-detected |
| `--experiment-name S` | MLflow experiment name | auto-generated |
| `--no-mock-data` | Skip mock data generation | false |

## 🔧 Troubleshooting

### Common Issues

**GPU Memory Error:**
```bash
./run_training.sh --batch-size 2  # Reduce batch size
```

**Python Package Missing:**
```bash
pip install torch transformers numpy pandas scikit-learn scipy PyYAML
```

**Permission Denied:**
```bash
chmod +x run_training.sh
```

### Debug Mode

For detailed debugging, check:
- `production_training.log` - Full execution log
- Console output during training
- MLflow experiment dashboard (if available)

## 🚀 Production Deployment

After successful training:

1. **Test the Model:**
   ```python
   # Load and test your model
   import torch
   checkpoint = torch.load('production_data/models/final_model/model.pt')
   ```

2. **Deploy to Databricks:**
   - Upload the inference package to DBFS
   - Create model serving endpoint
   - Set up monitoring

3. **Production Monitoring:**
   - Set up automated retraining
   - Monitor model performance
   - Track data drift

## 📚 Advanced Usage

For more advanced features, see:
- `DATABRICKS_PRODUCTION_GUIDE.md` - Detailed Databricks setup
- `databricks_training_notebook.py` - Interactive notebook
- `FIXES_SUMMARY.md` - Technical implementation details

## 🎉 Success!

If you see `🎉 TRAINING COMPLETED SUCCESSFULLY!`, your multimodal LLM is ready for production use!

---

**Need help?** Check the detailed guides or review the logs for specific error messages.