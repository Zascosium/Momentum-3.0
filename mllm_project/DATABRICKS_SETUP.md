# Databricks Setup Guide for MLLM Pipeline

This guide helps you set up the Multimodal LLM Pipeline on Databricks.

## 🚀 Quick Start

1. **Upload project files** to your Databricks workspace
2. **Install dependencies**:
   ```bash
   python install_databricks.py
   ```
3. **Test the setup**:
   ```bash
   python debug_imports.py
   ```
4. **Run your first command**:
   ```bash
   python cli.py explore --data-dir /path/to/your/data --sample-size 100
   ```

## 📋 Prerequisites

- Databricks Runtime 10.4 LTS or higher
- Python 3.8+
- Access to install packages (if using a restricted cluster, pre-install dependencies)

## 🔧 Detailed Setup

### Step 1: Upload Project Files

Upload all project files to your Databricks workspace. You can:
- Use the Databricks web interface
- Use Git integration (recommended)
- Use the Databricks CLI

### Step 2: Install Dependencies

Run the automated installer:

```bash
python install_databricks.py
```

Or install manually:

```bash
# Core dependencies
pip install click>=8.1.0 pyyaml>=6.0 rich>=13.3.0
pip install numpy>=1.21.0 pandas>=2.0.0 matplotlib>=3.7.0 seaborn>=0.12.0

# ML dependencies  
pip install torch>=2.0.0 transformers>=4.30.0 scikit-learn>=1.3.0 scipy>=1.10.0

# Optional but recommended
pip install omegaconf>=2.3.0 hydra-core>=1.3.0
pip install mlflow>=2.4.0  # For training
pip install fastapi>=0.95.0 uvicorn>=0.22.0  # For serving
```

### Step 3: Test Your Setup

```bash
python debug_imports.py
```

This will test:
- Environment detection
- Dependency availability  
- Pipeline imports
- CLI command functionality

### Step 4: Configure Data Paths

Update your data paths for Databricks:

```python
# Example: Using DBFS paths
python cli.py explore --data-dir /dbfs/FileStore/your-data --sample-size 1000
```

## 🐛 Troubleshooting

### Common Issues

#### "DataExplorationPipeline not available"

**Solution**: Install dependencies and check imports:
```bash
python install_databricks.py
python debug_imports.py
```

#### "No module named 'mlflow'"

**Solution**: Install MLflow for training features:
```bash
pip install mlflow>=2.4.0
```

#### "FastAPI not available"

**Solution**: Install FastAPI for serving features:
```bash
pip install fastapi uvicorn
```

#### Import errors with relative paths

This is handled automatically by the Databricks compatibility layer. If you still see issues:

1. Ensure you're running from the project root directory
2. Check that all files were uploaded correctly
3. Run the debug script for detailed diagnostics

### Databricks-Specific Considerations

1. **File Paths**: Use DBFS paths (`/dbfs/...`) for data files
2. **Dependencies**: Some packages may need cluster-level installation
3. **Memory**: Large models may require clusters with sufficient RAM
4. **Permissions**: Ensure you have write permissions for output directories

## ✅ Command Examples

Once setup is complete, you can use all CLI commands:

### Data Exploration
```bash
python cli.py explore --data-dir /dbfs/FileStore/data --sample-size 1000 --generate-report
```

### Training (requires MLflow)
```bash
python cli.py train --epochs 10 --batch-size 32 --mixed-precision
```

### Evaluation (requires trained model)
```bash
python cli.py evaluate --model-path /dbfs/FileStore/models/best_model.pt --generate-plots
```

### Interactive Demo (requires trained model)
```bash
python cli.py demo --model-path /dbfs/FileStore/models/best_model.pt --interactive
```

### Full Pipeline
```bash
python cli.py pipeline --config ./config/pipeline_config.yaml
```

### Model Serving (requires FastAPI)
```bash
python cli.py serve --model-path /dbfs/FileStore/models/best_model.pt --port 8080
```

## 📚 Additional Resources

- [Databricks Documentation](https://docs.databricks.com/)
- [MLflow on Databricks](https://docs.databricks.com/mlflow/index.html)
- [DBFS File System](https://docs.databricks.com/dbfs/index.html)

## 🆘 Getting Help

If you encounter issues:

1. Run `python debug_imports.py` for detailed diagnostics
2. Check the error messages for specific missing dependencies
3. Ensure all project files are uploaded correctly
4. Verify cluster permissions and resources

The pipeline is designed to be robust and provide clear error messages for missing dependencies, so you can install only what you need for your specific use case.