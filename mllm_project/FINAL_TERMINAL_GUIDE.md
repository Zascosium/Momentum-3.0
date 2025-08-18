# 🚀 **Complete Terminal Execution Guide for Multimodal LLM**

## 📋 **You Now Have 4 Different Ways to Run from Terminal!**

### **📋 Quick Reference**

| Method | Command | Use Case |
|--------|---------|----------|
| **Simple Script** | `./run_training.sh` | Easiest, one-command execution |
| **Full Pipeline** | `python run_pipeline.py full` | Complete pipeline with all features |
| **Standalone Training** | `python train_standalone.py` | Minimal, fast training only |
| **Databricks Jobs** | `python databricks_submit_job.py submit` | Cloud cluster submission |

---

## 🎯 **Method 1: Simple Bash Script (Easiest)**

### **One-Command Training**
```bash
# Quick test (2 minutes)
./run_training.sh --quick

# Default training (30 minutes)
./run_training.sh

# Full training (2-4 hours)
./run_training.sh --full

# Custom training
./run_training.sh --domains "Agriculture Climate" --epochs 5 --batch-size 4
```

### **All Options**
```bash
./run_training.sh --help              # Show help
./run_training.sh --quick             # Quick training (1 epoch)
./run_training.sh --full              # Full training (all domains, 5 epochs)
./run_training.sh --method standalone # Use standalone trainer
```

---

## 🔧 **Method 2: Full Pipeline Runner**

### **Complete Pipeline**
```bash
# Validate environment
python run_pipeline.py validate

# Full pipeline (recommended)
python run_pipeline.py full --domains Agriculture Climate --epochs 3 --batch-size 2

# Step-by-step execution
python run_pipeline.py explore --domains Agriculture Climate
python run_pipeline.py train --domains Agriculture Climate --epochs 5 --batch-size 4
python run_pipeline.py evaluate
python run_pipeline.py infer --num-samples 10
```

### **Advanced Options**
```bash
# Skip exploration and evaluation (faster)
python run_pipeline.py full --domains Agriculture --epochs 3 --skip-exploration --skip-evaluation

# Debug mode
python run_pipeline.py full --domains Agriculture --epochs 1 --log-level DEBUG

# Custom learning rate
python run_pipeline.py full --domains Climate --epochs 3 --learning-rate 1e-4
```

---

## 🏋️ **Method 3: Standalone Training Script**

### **Pure Training (No Pipeline Overhead)**
```bash
# Basic training
python train_standalone.py --domains Agriculture --epochs 3 --batch-size 2

# High-performance training
python train_standalone.py --domains Agriculture Climate Economy --epochs 10 --batch-size 8

# Quick test
python train_standalone.py --domains Agriculture --epochs 1 --batch-size 1
```

### **Features**
- ✅ **Fastest startup** - Minimal overhead
- ✅ **Progress bars** - Real-time training progress
- ✅ **Automatic saving** - Best model checkpointing
- ✅ **Detailed logs** - Complete training history

---

## ☁️ **Method 4: Databricks Job Submission**

### **Submit to Cloud Clusters**
```bash
# Setup (one-time)
pip install databricks-cli
databricks configure --token

# Submit training job
python databricks_submit_job.py submit \
    --domains Agriculture Climate \
    --epochs 5 \
    --batch-size 4 \
    --upload \
    --monitor

# Monitor existing job
python databricks_submit_job.py monitor --run-id 12345

# List recent jobs
python databricks_submit_job.py list
```

### **Job Management**
```bash
# Submit and detach (background)
python databricks_submit_job.py submit --domains Economy --epochs 3

# Upload project only
python databricks_submit_job.py submit --upload --domains Agriculture --epochs 1
```

---

## 📊 **Expected Results from All Methods**

After successful execution, you'll have:

### **🤖 Trained Model**
```
checkpoints/
├── best_model.pt           # Best performing model
├── model.pt               # Final model state
└── checkpoint_epoch_N.pt  # Training checkpoints
```

### **📈 Results & Metrics**
```
outputs/
├── training_summary.json     # Training overview
├── training_history.json     # Loss curves
├── evaluation_results.json   # Model performance
└── demo_results.json        # Sample predictions
```

### **📋 Logs**
```
logs/
├── pipeline.log             # Full pipeline logs
├── standalone_training.log  # Standalone training logs
└── training.log            # General training logs
```

---

## ⚡ **Quick Start Examples**

### **For Testing (5 minutes)**
```bash
# Option 1: Simple script
./run_training.sh --quick

# Option 2: Direct Python
python train_standalone.py --domains Agriculture --epochs 1 --batch-size 1
```

### **For Development (30 minutes)**
```bash
# Option 1: Simple script  
./run_training.sh --domains "Agriculture Climate" --epochs 3

# Option 2: Full pipeline
python run_pipeline.py full --domains Agriculture Climate --epochs 3 --batch-size 2
```

### **For Production (2-4 hours)**
```bash
# Databricks cluster
python databricks_submit_job.py submit \
    --domains Agriculture Climate Economy \
    --epochs 10 \
    --batch-size 4 \
    --upload --monitor
```

---

## 🔍 **Monitoring & Troubleshooting**

### **Monitor Training**
```bash
# Watch logs in real-time
tail -f logs/pipeline.log
tail -f logs/standalone_training.log

# Check GPU usage
nvidia-smi -l 1

# Monitor disk usage
df -h
```

### **Common Issues & Solutions**

1. **CUDA Out of Memory**:
```bash
# Reduce batch size
python run_pipeline.py full --domains Agriculture --batch-size 1
```

2. **Module Import Errors**:
```bash
# Fix Python path
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

3. **Missing Dependencies**:
```bash
# Install requirements
pip install torch torchvision transformers accelerate datasets pyyaml tqdm matplotlib
```

4. **Permission Denied**:
```bash
# Make script executable
chmod +x run_training.sh
```

---

## 📊 **Performance Comparison**

| Method | Setup Time | Flexibility | Resource Control | Monitoring |
|--------|------------|-------------|------------------|------------|
| **Pipeline Runner** | Fast | High | Medium | Good |
| **Standalone Script** | Fastest | Medium | High | Excellent |
| **Databricks Jobs** | Medium | Medium | Low | Good |

### **Recommendations**

- **Development/Testing**: Use `train_standalone.py`
- **Production Training**: Use `run_pipeline.py full`
- **Cloud Training**: Use Databricks job submission

---

## 🎯 **Choose Your Method**

| **Scenario** | **Recommended Method** | **Command** |
|--------------|------------------------|-------------|
| **First time / Testing** | Simple Script | `./run_training.sh --quick` |
| **Development** | Full Pipeline | `python run_pipeline.py full` |
| **Fast Training Only** | Standalone | `python train_standalone.py` |
| **Cloud Training** | Databricks Jobs | `python databricks_submit_job.py submit` |

---

## 🎯 **Complete Example Workflows**

### **Quick Test Run (2 minutes)**
```bash
# Validate environment
python run_pipeline.py validate

# Quick training test
python train_standalone.py --domains Agriculture --epochs 1 --batch-size 1

# Check results
ls checkpoints/ outputs/
```

### **Development Workflow (30 minutes)**
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

### **Production Training (2-4 hours)**
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

## 🎉 **Summary**

You now have **4 different ways** to run the multimodal LLM pipeline from the terminal:

1. **`./run_training.sh`** - Simple one-command execution
2. **`run_pipeline.py`** - Full pipeline with all features
3. **`train_standalone.py`** - Minimal, fast training script  
4. **`databricks_submit_job.py`** - Cloud cluster job submission

### **Choose the method that best fits your needs:**
- **Quick testing**: `./run_training.sh --quick` or `train_standalone.py`
- **Complete pipeline**: `run_pipeline.py full`
- **Cloud training**: Databricks job submission

All methods produce the same trained multimodal LLM capable of processing time series data and generating text analyses! 🤖✨

### **Key Features Across All Methods:**
- ✅ **MOMENT encoder** + **Cross-attention** + **GPT-2 decoder**
- ✅ **Time-MMD dataset** support (Agriculture, Climate, Economy)
- ✅ **Automatic checkpointing** and result saving
- ✅ **Real-time monitoring** and logging
- ✅ **Production-ready** multimodal LLM output

**Ready to start training? Pick your method and go!** 🚀