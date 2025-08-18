# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 Multimodal LLM - Complete Databricks Setup & Execution
# MAGIC 
# MAGIC This notebook provides a complete, one-click solution to run the multimodal LLM pipeline in Databricks.
# MAGIC 
# MAGIC **What this does:**
# MAGIC - ✅ Environment validation
# MAGIC - 🔧 Automatic dependency installation  
# MAGIC - 📊 Data exploration
# MAGIC - 🏋️ Model training (MOMENT + GPT-2)
# MAGIC - 📈 Model evaluation
# MAGIC - 🎯 Inference demonstration
# MAGIC 
# MAGIC **Requirements:**
# MAGIC - Databricks Runtime 13.3 LTS ML (GPU-enabled)
# MAGIC - Cluster with 16+ GB RAM, GPU support
# MAGIC - Project uploaded to Workspace

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏗️ Step 1: Environment Setup & Validation

# COMMAND ----------

import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("🔍 DATABRICKS ENVIRONMENT CHECK")
print("=" * 50)

# Detect project location
possible_locations = [
    "/Workspace/Repos/mllm_project",
    f"/Workspace/Users/{os.environ.get('DB_USER_NAMESPACE', 'user@company.com')}/mllm_project",
    "/databricks/driver/mllm_project"
]

project_root = None
for location in possible_locations:
    if os.path.exists(location):
        project_root = location
        break

if project_root is None:
    print("❌ Project not found! Please upload the mllm_project folder to your workspace.")
    print("Expected locations:")
    for loc in possible_locations:
        print(f"  - {loc}")
    raise FileNotFoundError("Project not found in workspace")

print(f"✅ Project found: {project_root}")

# Setup Python paths
sys.path.insert(0, f"{project_root}/src")
sys.path.insert(0, project_root)

# Environment info
print(f"📍 Databricks Runtime: {os.environ.get('DATABRICKS_RUNTIME_VERSION', 'Unknown')}")
print(f"🖥️  Cluster ID: {os.environ.get('DB_CLUSTER_ID', 'Unknown')}")
print(f"👤 User: {os.environ.get('DB_USER_NAMESPACE', 'Unknown')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Step 2: Install Dependencies

# COMMAND ----------

# Install required packages
print("📦 Installing dependencies...")

# Core ML packages
%pip install torch==2.0.0 torchvision==0.15.0 --quiet
%pip install transformers==4.35.0 accelerate==0.24.0 --quiet
%pip install datasets==2.14.0 --quiet

# Utilities
%pip install pyyaml==6.0 tqdm==4.65.0 --quiet
%pip install matplotlib==3.7.0 seaborn==0.12.0 --quiet

# MLflow is pre-installed in Databricks ML Runtime
print("✅ All dependencies installed!")

# Restart Python to ensure clean imports
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚀 Step 3: Initialize Pipeline

# COMMAND ----------

# Re-setup after restart
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Re-detect project (after restart)
possible_locations = [
    "/Workspace/Repos/mllm_project",
    f"/Workspace/Users/{os.environ.get('DB_USER_NAMESPACE', 'user@company.com')}/mllm_project",
    "/databricks/driver/mllm_project"
]

project_root = None
for location in possible_locations:
    if os.path.exists(location):
        project_root = location
        break

# Re-setup paths
sys.path.insert(0, f"{project_root}/src")
sys.path.insert(0, project_root)

# Initialize pipeline
from databricks_pipeline import DatabricksPipeline

print("🔧 Initializing Multimodal LLM Pipeline...")
pipeline = DatabricksPipeline(project_root)

print("✅ Pipeline initialized successfully!")
print(f"📁 Project Root: {pipeline.project_root}")
print(f"💾 Checkpoints: {pipeline.checkpoint_dir}")
print(f"📊 Outputs: {pipeline.output_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 Step 4: Environment Validation

# COMMAND ----------

# Validate environment
print("🔍 VALIDATING ENVIRONMENT")
print("=" * 40)

validation_results = pipeline.validate_environment()

# Display results nicely
import json
print(json.dumps(validation_results, indent=2))

if validation_results['status'] == 'success':
    print("\n✅ Environment validation PASSED!")
    if validation_results.get('cuda_available'):
        print(f"🚀 GPU Available: {validation_results['gpu_name']}")
        print(f"💾 GPU Memory: {validation_results['gpu_memory_gb']:.1f} GB")
else:
    print("\n❌ Environment validation FAILED!")
    if validation_results.get('missing_modules'):
        print(f"Missing modules: {validation_results['missing_modules']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 Step 5: Data Exploration (Optional)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run this cell if you want to explore the data first

# COMMAND ----------

# Explore the Time-MMD dataset
print("📊 EXPLORING TIME-MMD DATASET")
print("=" * 40)

try:
    exploration_results = pipeline.run_data_exploration()
    
    # Display key statistics
    if 'domain_stats' in exploration_results:
        print("\n📈 Dataset Statistics:")
        for domain, stats in exploration_results['domain_stats'].items():
            print(f"  {domain}: {stats.get('total_samples', 'Unknown')} samples")
    
    if 'data_quality' in exploration_results:
        print(f"\n✅ Data Quality Score: {exploration_results['data_quality'].get('overall_score', 'Unknown')}")
    
    print(f"\n📁 Detailed results saved to: {pipeline.output_dir}/exploration_results.json")
    
except Exception as e:
    print(f"⚠️ Data exploration failed: {e}")
    print("Continuing with training...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏋️ Step 6: Model Training

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Training Parameters
# MAGIC 
# MAGIC Adjust these parameters based on your cluster resources:

# COMMAND ----------

# Training configuration
DOMAINS = ["Agriculture", "Climate", "Economy"]  # Domains to train on
EPOCHS = 3                                       # Number of training epochs  
BATCH_SIZE = 2                                   # Batch size (reduce if OOM)

print(f"🏋️ TRAINING CONFIGURATION")
print(f"Domains: {DOMAINS}")
print(f"Epochs: {EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print("=" * 40)

# Start training
print("🚀 Starting training...")
training_results = pipeline.run_training(
    domains=DOMAINS,
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE
)

print("✅ Training completed!")
print(f"📊 Final Training Loss: {training_results.get('final_metrics', {}).get('train_loss', 'Unknown')}")
print(f"📊 Final Validation Loss: {training_results.get('final_metrics', {}).get('val_loss', 'Unknown')}")
print(f"🤖 Model saved to: {training_results['model_path']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 Step 7: Model Evaluation

# COMMAND ----------

print("📈 EVALUATING TRAINED MODEL")
print("=" * 40)

# Run evaluation
evaluation_results = pipeline.run_evaluation()

print("✅ Evaluation completed!")

# Display key metrics
if 'metrics' in evaluation_results:
    metrics = evaluation_results['metrics']
    print(f"\n📊 Model Performance:")
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric_name}: {value:.4f}")

print(f"\n📁 Detailed results saved to: {pipeline.output_dir}/evaluation_results.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Step 8: Inference Demonstration

# COMMAND ----------

print("🎯 RUNNING INFERENCE DEMO")
print("=" * 40)

# Run inference demo
demo_results = pipeline.run_inference_demo(num_samples=3)

print("✅ Inference demo completed!")

# Display sample predictions
if 'predictions' in demo_results:
    print("\n🔮 Sample Predictions:")
    for i, prediction in enumerate(demo_results['predictions'][:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Domain: {prediction.get('domain', 'Unknown')}")
        print(f"Generated Text: {prediction.get('generated_text', 'No text generated')[:200]}...")

print(f"\n📁 All demo results saved to: {pipeline.output_dir}/demo_results.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎉 Step 9: Pipeline Summary

# COMMAND ----------

print("🎉 MULTIMODAL LLM PIPELINE COMPLETED!")
print("=" * 50)

# Get final status
status = pipeline.get_status()

print(f"📍 Project Location: {status['project_root']}")
print(f"💾 Checkpoints: {status['checkpoint_dir']}")
print(f"📊 Results: {status['output_dir']}")

print(f"\n🤖 Available Models:")
for model in status['available_models']:
    print(f"  - {model['name']} ({model['size_mb']:.1f} MB)")

print(f"\n📋 Available Results:")
for result in status['available_results']:
    print(f"  - {result['name']}")

print("\n✅ Your multimodal LLM is ready for use!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔧 Step 10: Quick Testing (Optional)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the trained model with a quick prediction

# COMMAND ----------

print("🧪 QUICK MODEL TEST")
print("=" * 30)

try:
    # Load the trained model
    import torch
    from models.multimodal_model import MultimodalLLM
    
    model_path = f"{pipeline.checkpoint_dir}/best_model.pt"
    print(f"Loading model from: {model_path}")
    
    # Check if model exists
    if os.path.exists(model_path):
        # Load model (simplified for demo)
        print("✅ Model file found!")
        print("🔮 Model is ready for inference!")
        
        # You can add actual inference code here
        print("\n💡 To use the model in production:")
        print("1. Load the model using MultimodalLLM.load_pretrained()")
        print("2. Prepare time series data and text prompts")
        print("3. Call model.generate() for text generation")
        
    else:
        print("❌ Model not found. Please run training first.")
        
except Exception as e:
    print(f"⚠️ Test failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📚 Next Steps
# MAGIC 
# MAGIC **Your multimodal LLM is now trained and ready!**
# MAGIC 
# MAGIC ### What you can do next:
# MAGIC 
# MAGIC 1. **Production Deployment**:
# MAGIC    - Register model in MLflow Model Registry
# MAGIC    - Create model serving endpoint
# MAGIC    - Set up batch inference jobs
# MAGIC 
# MAGIC 2. **Model Improvement**:
# MAGIC    - Increase training epochs for better performance
# MAGIC    - Add more domains to the training data
# MAGIC    - Fine-tune hyperparameters
# MAGIC 
# MAGIC 3. **Integration**:
# MAGIC    - Create REST API endpoints
# MAGIC    - Build web interface for predictions
# MAGIC    - Connect to real-time data sources
# MAGIC 
# MAGIC ### Key Files Created:
# MAGIC - **Model**: `/dbfs/mllm_checkpoints/best_model.pt`
# MAGIC - **Results**: `/dbfs/mllm_outputs/`
# MAGIC - **Logs**: Available in MLflow experiments
# MAGIC 
# MAGIC ### Support:
# MAGIC - Check `/dbfs/mllm_outputs/` for detailed results
# MAGIC - Review MLflow experiments for training metrics
# MAGIC - Examine logs for any issues
# MAGIC 
# MAGIC **🎉 Congratulations! You've successfully deployed a multimodal LLM in Databricks!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 One-Click Full Pipeline (Alternative)
# MAGIC 
# MAGIC ### If you want to run everything in one go, use this cell instead of the step-by-step approach above:

# COMMAND ----------

# # Uncomment and run this for a complete one-shot execution:
# 
# # from databricks_pipeline import DatabricksPipeline
# # 
# # # Run complete pipeline
# # pipeline = DatabricksPipeline(project_root)
# # 
# # results = pipeline.run_full_pipeline(
# #     domains=["Agriculture", "Climate"],
# #     epochs=3,
# #     batch_size=2,
# #     skip_exploration=True
# # )
# # 
# # print("🎉 Complete pipeline finished!")
# # print(f"Status: {results['status']}")
# # print(f"Model: {results['model_location']}")

# COMMAND ----------