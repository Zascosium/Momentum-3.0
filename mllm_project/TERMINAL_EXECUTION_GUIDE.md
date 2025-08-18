# 🚀 Terminal-Based Execution Guide for Multimodal LLM

This guide shows you **multiple ways to run the multimodal LLM pipeline from the terminal** instead of using notebooks.

## 📋 Quick Start Options

### Option 1: Full Pipeline Runner (Recommended)
```bash
python run_pipeline.py full --domains Agriculture Climate --epochs 3 --batch-size 2
```

### Option 2: Standalone Training Script
```bash
python train_standalone.py --domains Agriculture --epochs 3 --batch-size 2
```

### Option 3: Databricks Job Submission
```bash
python databricks_submit_job.py submit --domains Agriculture --epochs 3 --upload --monitor
```

---

## 🔧 Method 1: Full Pipeline Runner

### Basic Usage
```bash
# Validate environment only
python run_pipeline.py validate

# Quick training (minimal resources)
python run_pipeline.py full --domains Agriculture --epochs 2 --batch-size 1

# Full training (recommended)
python run_pipeline.py full --domains Agriculture Climate Economy --epochs 5 --batch-size 4

# Training with custom learning rate
python run_pipeline.py full --domains Climate --epochs 3 --learning-rate 1e-4
```

### Step-by-Step Execution
```bash
# 1. Validate environment
python run_pipeline.py validate

# 2. Explore data (optional)
python run_pipeline.py explore --domains Agriculture Climate

# 3. Train model
python run_pipeline.py train --domains Agriculture Climate --epochs 5 --batch-size 4

# 4. Evaluate model
python run_pipeline.py evaluate

# 5. Run inference demo
python run_pipeline.py infer --num-samples 10
```

### Advanced Options
```bash
# Skip time-consuming steps
python run_pipeline.py full --domains Agriculture --epochs 3 --skip-exploration --skip-evaluation

# Custom project root
python run_pipeline.py full --project-root /path/to/mllm_project --domains Climate --epochs 3

# Debug mode
python run_pipeline.py full --domains Agriculture --epochs 1 --log-level DEBUG
```

---

## 🏋️ Method 2: Standalone Training Script

This is a **self-contained training script** that doesn't require the full pipeline infrastructure.

### Basic Training
```bash
# Minimal training
python train_standalone.py --domains Agriculture --epochs 2 --batch-size 1

# Full training
python train_standalone.py --domains Agriculture Climate Economy --epochs 5 --batch-size 4

# High-performance training
python train_standalone.py --domains Climate --epochs 10 --batch-size 8 --learning-rate 1e-4
```

### Features
- ✅ **Self-contained**: Minimal dependencies
- ✅ **Progress bars**: Real-time training progress
- ✅ **Automatic checkpointing**: Saves best models
- ✅ **Detailed logging**: Complete training logs
- ✅ **Resource monitoring**: GPU utilization tracking

### Output Files
```
logs/standalone_training.log    # Detailed training log
checkpoints/best_model.pt       # Best model checkpoint
outputs/training_summary.json   # Training summary
outputs/training_history.json   # Loss curves and metrics
```

---

## ☁️ Method 3: Databricks Job Submission

Submit training jobs to **Databricks clusters** from your terminal.

### Prerequisites
```bash
# Install Databricks CLI
pip install databricks-cli

# Configure authentication
databricks configure --token
```

### Job Submission
```bash
# Submit job with project upload
python databricks_submit_job.py submit \
    --domains Agriculture Climate \
    --epochs 5 \
    --batch-size 4 \
    --upload \
    --monitor

# Submit job (project already uploaded)
python databricks_submit_job.py submit \
    --domains Economy \
    --epochs 3 \
    --project-path /Workspace/Users/you@company.com/mllm_project

# Monitor existing job
python databricks_submit_job.py monitor --run-id 12345

# List recent jobs
python databricks_submit_job.py list
```

### Job Management
```bash
# Submit and detach (run in background)
python databricks_submit_job.py submit --domains Agriculture --epochs 5

# Submit with custom cluster settings
python databricks_submit_job.py submit \
    --domains Climate Economy \
    --epochs 10 \
    --batch-size 8 \
    --learning-rate 1e-4
```

---

---

## 🔍 Monitoring & Results

### Real-Time Monitoring
```bash
# Monitor training logs
tail -f logs/pipeline.log

# Monitor standalone training
tail -f logs/standalone_training.log

# Monitor GPU usage
nvidia-smi -l 1
```

### Check Results
```bash
# List generated files
ls -la checkpoints/
ls -la outputs/

# View training summary
cat outputs/training_summary.json | python -m json.tool

# View final metrics
grep "Final" logs/pipeline.log
```

---

## ⚙️ Configuration Options

### Environment Variables
```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Set project root
export MLLM_PROJECT_ROOT=/path/to/project

# Set log level
export MLLM_LOG_LEVEL=DEBUG
```

### Resource Configuration
```bash
# For limited GPU memory (reduce batch size)
python run_pipeline.py full --domains Agriculture --epochs 3 --batch-size 1

# For high-memory GPUs (increase batch size)  
python run_pipeline.py full --domains Agriculture Climate Economy --epochs 5 --batch-size 8

# For CPU-only training (slower)
CUDA_VISIBLE_DEVICES="" python run_pipeline.py full --domains Agriculture --epochs 2
```

---

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
```bash
# Reduce batch size
python run_pipeline.py full --domains Agriculture --epochs 3 --batch-size 1
```

2. **Module Import Errors**:
```bash
# Check Python path
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python run_pipeline.py validate
```

3. **Missing Dependencies**:
```bash
# Install requirements
pip install torch torchvision transformers accelerate datasets pyyaml tqdm matplotlib seaborn
```

4. **Databricks Authentication**:
```bash
# Reconfigure Databricks CLI
databricks configure --token

# Test connection
databricks workspace list
```

### Debug Mode
```bash
# Enable debug logging for any method
python run_pipeline.py full --domains Agriculture --epochs 1 --log-level DEBUG

# Standalone debug
python train_standalone.py --domains Agriculture --epochs 1 --log-level DEBUG
```

---

## 📊 Performance Comparison

| Method | Setup Time | Flexibility | Resource Control | Monitoring |
|--------|------------|-------------|------------------|------------|
| **Pipeline Runner** | Fast | High | Medium | Good |
| **Standalone Script** | Fastest | Medium | High | Excellent |
| **Databricks Jobs** | Medium | Medium | Low | Good |

### Recommendations

- **Development/Testing**: Use `train_standalone.py`
- **Production Training**: Use `run_pipeline.py full`
- **Cloud Training**: Use Databricks job submission

---

## 🎯 Complete Example Workflows

### Quick Test Run (2 minutes)
```bash
# Validate environment
python run_pipeline.py validate

# Quick training test
python train_standalone.py --domains Agriculture --epochs 1 --batch-size 1

# Check results
ls checkpoints/ outputs/
```

### Development Workflow (30 minutes)
```bash
# Full validation
python run_pipeline.py validate

# Data exploration
python run_pipeline.py explore --domains Agriculture Climate

# Training with checkpoints
python run_pipeline.py train --domains Agriculture Climate --epochs 3 --batch-size 2

# Evaluate and demo
python run_pipeline.py evaluate
python run_pipeline.py infer --num-samples 5
```

### Production Training (2-4 hours)
```bash
# Submit to Databricks cluster
python databricks_submit_job.py submit \
    --domains Agriculture Climate Economy \
    --epochs 10 \
    --batch-size 4 \
    --learning-rate 5e-5 \
    --upload \
    --monitor
```

---

## 🎉 Summary

You now have **3 different ways** to run the multimodal LLM pipeline from the terminal:

1. **`run_pipeline.py`** - Full pipeline with all features
2. **`train_standalone.py`** - Minimal, fast training script  
3. **`databricks_submit_job.py`** - Cloud cluster job submission

Choose the method that best fits your needs:
- **Quick testing**: `train_standalone.py`
- **Complete pipeline**: `run_pipeline.py full`
- **Cloud training**: Databricks job submission

All methods produce the same trained multimodal LLM capable of processing time series data and generating text analyses! 🤖✨