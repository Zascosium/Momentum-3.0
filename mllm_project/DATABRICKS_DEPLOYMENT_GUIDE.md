# Databricks Deployment Guide for Multimodal LLM Pipeline

This guide will walk you through deploying and running the complete multimodal LLM pipeline in a Databricks workspace.

## 📋 Prerequisites

1. **Databricks Workspace** with ML Runtime 13.3 LTS or newer
2. **Cluster Configuration**:
   - Runtime: ML Runtime 13.3 LTS (Scala 2.12, Spark 3.4.0)
   - Python Version: 3.10+
   - Node Type: Single Node with GPU (e.g., g4dn.xlarge or better)
   - Driver: 16+ GB RAM, 4+ cores
   - Workers: Same as driver for single-node setup

## 🚀 Step 1: Upload Project to Databricks

### Option A: Using Databricks CLI (Recommended)

1. Install Databricks CLI locally:
```bash
pip install databricks-cli
```

2. Configure authentication:
```bash
databricks configure --token
```

3. Upload the entire project:
```bash
databricks workspace import-dir ./mllm_project /Workspace/Users/{your-email}/mllm_project --overwrite
```

### Option B: Manual Upload via Web UI

1. Compress the project:
```bash
cd mllm_project
tar -czf mllm_project.tar.gz *
```

2. Upload via Databricks UI:
   - Go to Workspace → Users → {your-email}
   - Create folder: `mllm_project`
   - Upload and extract the archive

## 🏗️ Step 2: Set Up Cluster

1. **Create New Cluster**:
   ```
   Cluster Name: mllm-training-cluster
   Databricks Runtime Version: 13.3 LTS ML (includes Apache Spark 3.4.0, GPU, Scala 2.12)
   Node Type: g4dn.xlarge (or similar GPU instance)
   Min Workers: 0
   Max Workers: 0 (Single Node)
   ```

2. **Advanced Options** → **Environment Variables**:
   ```
   PYTHONPATH=/Workspace/Users/{your-email}/mllm_project/src
   CUDA_VISIBLE_DEVICES=0
   ```

3. **Libraries** (Install these via Cluster UI):
   - PyPI: `transformers==4.35.0`
   - PyPI: `torch==2.0.0`
   - PyPI: `torchvision==0.15.0`
   - PyPI: `accelerate==0.24.0`
   - PyPI: `datasets==2.14.0`
   - PyPI: `mlflow==2.7.0`
   - PyPI: `pyyaml==6.0`
   - PyPI: `numpy==1.24.0`
   - PyPI: `pandas==2.0.0`
   - PyPI: `tqdm==4.65.0`
   - PyPI: `matplotlib==3.7.0`
   - PyPI: `seaborn==0.12.0`

## 📁 Step 3: File Structure Overview

After upload, your Databricks workspace should have:

```
/Workspace/Users/{your-email}/mllm_project/
├── config/                          # Configuration files
│   ├── data_config.yaml
│   ├── model_config.yaml
│   ├── pipeline_config.yaml
│   └── training_config.yaml
├── data/
│   └── time_mmd/                   # Time-MMD dataset
│       ├── dataset_info.json
│       ├── numerical/
│       └── textual/
├── src/                            # Source code
│   ├── data/                      # Data loading and preprocessing
│   ├── models/                    # Model implementations
│   ├── pipelines/                 # Pipeline implementations
│   ├── training/                  # Training components
│   └── utils/                     # Utilities
├── databricks_cli.py              # Databricks command interface
├── databricks_training_notebook.py # Main training notebook
└── checkpoints/                   # Model checkpoints (optional)
```

## 🔧 Step 4: Configure for Databricks

1. **Update Configuration Files** (Important!):

Edit `/Workspace/Users/{your-email}/mllm_project/config/pipeline_config.yaml`:
```yaml
environment: "databricks"
checkpoint_dir: "/dbfs/mllm_checkpoints"
data_dir: "/Workspace/Users/{your-email}/mllm_project/data/time_mmd"
output_dir: "/dbfs/mllm_outputs"
```

Edit `/Workspace/Users/{your-email}/mllm_project/config/training_config.yaml`:
```yaml
mixed_precision: true
device: "cuda"
num_workers: 0  # Important for Databricks
persistent_workers: false
pin_memory: true
```

## 🏃‍♂️ Step 5: Run the Pipeline

### Method 1: Using the Training Notebook (Recommended)

1. **Open the Training Notebook**:
   - Navigate to: `/Workspace/Users/{your-email}/mllm_project/databricks_training_notebook.py`
   - Open it as a Databricks Notebook

2. **Attach to Cluster**:
   - Click "Connect" and select your `mllm-training-cluster`

3. **Run All Cells**:
   - Click "Run All" to execute the complete pipeline
   - Monitor progress in real-time

### Method 2: Using CLI Interface

1. **Create a New Notebook** and add:

```python
# Cell 1: Setup
import sys
import os
sys.path.insert(0, "/Workspace/Users/{your-email}/mllm_project/src")

# Cell 2: Import and Initialize
from databricks_cli import DatabricksCLI

cli = DatabricksCLI()
cli.setup_environment()

# Cell 3: Run Full Pipeline
results = cli.run_training_pipeline(
    domains=["Agriculture", "Climate", "Economy"],
    epochs=5,
    batch_size=4
)

print("Training Results:", results)
```

### Method 3: Step-by-Step Execution

1. **Data Exploration**:
```python
from pipelines.exploration_pipeline import DataExplorationPipeline

pipeline = DataExplorationPipeline()
exploration_results = pipeline.run()
pipeline.display_results()
```

2. **Training**:
```python
from pipelines.training_pipeline_databricks import DatabricksTrainingPipeline

training_pipeline = DatabricksTrainingPipeline()
model, metrics = training_pipeline.run(
    domains=["Agriculture", "Climate", "Economy"],
    epochs=5
)
```

3. **Evaluation**:
```python
from pipelines.evaluation_pipeline import EvaluationPipeline

eval_pipeline = EvaluationPipeline()
eval_results = eval_pipeline.run(model)
```

4. **Inference Demo**:
```python
from pipelines.demo_pipeline_databricks import DatabricksDemoPipeline

demo_pipeline = DatabricksDemoPipeline()
demo_results = demo_pipeline.run(model)
```

## 📊 Step 6: Monitor Training

### MLflow Tracking

1. **View Experiments**:
   - Go to "Machine Learning" → "Experiments" in Databricks UI
   - Find experiment: `/Users/{your-email}/mllm_training`

2. **Monitor Metrics**:
   - Training loss
   - Validation loss  
   - Learning rate
   - Model parameters

### Real-time Monitoring

```python
# Add this to any notebook cell for real-time monitoring
import mlflow

# Get current experiment
experiment = mlflow.get_experiment_by_name("/Users/{your-email}/mllm_training")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Display latest metrics
print("Latest Training Metrics:")
print(runs[['metrics.train_loss', 'metrics.val_loss', 'status']].head())
```

## 💾 Step 7: Model Persistence

### Automatic Checkpointing

The pipeline automatically saves models to DBFS:
- **Location**: `/dbfs/mllm_checkpoints/`
- **Files**: 
  - `best_model.pt` - Best validation model
  - `final_model.pt` - Final epoch model
  - `training_summary.json` - Training metadata

### Manual Model Saving

```python
# Save trained model
model_path = "/dbfs/mllm_checkpoints/my_model"
model.save_pretrained(model_path)

# Load saved model later
from models.multimodal_model import MultimodalLLM
loaded_model = MultimodalLLM.load_pretrained(model_path)
```

## 🔍 Step 8: Inference and Demo

### Quick Inference Test

```python
# Load your trained model
model_path = "/dbfs/mllm_checkpoints/best_model.pt"
model = torch.load(model_path)
model.eval()

# Sample inference
import torch
time_series = torch.randn(1, 100, 1)  # Sample time series
text_prompt = "Analyze this time series data:"

# Generate analysis
generated_text = model.generate(
    time_series=time_series,
    text_input_ids=tokenize_text(text_prompt),
    max_length=100
)
print("Generated Analysis:", generated_text)
```

### Demo Pipeline

```python
from pipelines.demo_pipeline_databricks import DatabricksDemoPipeline

demo = DatabricksDemoPipeline()
demo.load_model("/dbfs/mllm_checkpoints/best_model.pt")

# Run demo with sample data
demo_results = demo.run_demo(
    domain="Agriculture",
    num_samples=5
)

demo.display_results(demo_results)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
```python
# Add to notebook cell
import sys
sys.path.insert(0, "/Workspace/Users/{your-email}/mllm_project/src")
sys.path.insert(0, "/Workspace/Users/{your-email}/mllm_project")
```

2. **GPU Not Detected**:
```python
# Check GPU availability
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

3. **Memory Issues**:
```python
# Reduce batch size in config
training_config['batch_size'] = 2
training_config['gradient_accumulation_steps'] = 4
```

4. **Library Conflicts**:
```bash
# Restart cluster and reinstall libraries in correct order
%pip install --upgrade torch torchvision
%pip install --upgrade transformers
%pip install --upgrade accelerate
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug flag
from databricks_cli import DatabricksCLI
cli = DatabricksCLI(debug=True)
```

## 🎯 Expected Outcomes

After successful execution, you should have:

1. **Trained Model**: Multimodal LLM capable of processing time series + text
2. **Evaluation Metrics**: Performance scores and visualizations  
3. **Model Artifacts**: Saved checkpoints in DBFS
4. **MLflow Experiment**: Complete training run logged
5. **Demo Results**: Sample predictions and generated text

### Success Metrics

- **Training Convergence**: Loss decreasing over epochs
- **Validation Performance**: Reasonable BLEU/ROUGE scores
- **Generation Quality**: Coherent text outputs
- **Memory Usage**: Under cluster limits
- **Runtime**: Complete pipeline in 30-60 minutes

## 🔄 Production Deployment

### Model Serving

1. **Register Model in MLflow**:
```python
import mlflow.pytorch

mlflow.pytorch.log_model(
    model,
    "multimodal_llm",
    registered_model_name="TimeSeries_TextGenerator"
)
```

2. **Create Model Endpoint**:
   - Go to "Machine Learning" → "Model Serving"
   - Create endpoint for registered model
   - Configure scaling and compute

### Batch Inference

```python
from pipelines.serving import BatchInferencePipeline

batch_pipeline = BatchInferencePipeline()
batch_pipeline.load_model("/dbfs/mllm_checkpoints/best_model.pt")

# Process large datasets
results = batch_pipeline.process_batch(
    input_data_path="/dbfs/input_data/",
    output_path="/dbfs/predictions/"
)
```

## 📚 Additional Resources

- [Databricks ML Runtime Documentation](https://docs.databricks.com/runtime/mlruntime.html)
- [MLflow on Databricks](https://docs.databricks.com/applications/mlflow/index.html)
- [GPU Clusters on Databricks](https://docs.databricks.com/compute/gpu.html)

---

## 🎉 Congratulations!

You now have a fully functional multimodal LLM pipeline running in Databricks. The system can process time series data with textual context and generate meaningful insights and predictions.

For any issues, check the troubleshooting section or examine the logs in the notebook outputs and MLflow experiments.