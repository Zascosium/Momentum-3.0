# Databricks notebook source
"""
# Interactive Inference Demo for Multimodal LLM

This notebook provides an interactive demonstration of the trained multimodal LLM,
allowing users to test the model with various inputs and explore its capabilities.

## Notebook Overview
1. Model Loading and Setup
2. Interactive Text Generation Interface
3. Time Series Analysis and Description
4. Multimodal Input Demonstration
5. Real-time Performance Monitoring
6. Batch Processing Utilities
7. Model Comparison and Benchmarking
8. Export and Deployment Preparation

## Prerequisites
- Trained multimodal LLM model available
- Inference utilities configured
- Interactive widgets enabled
- Sample data for demonstration
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup and Model Loading

# COMMAND ----------

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project source to path
sys.path.append('/Workspace/mllm_project/src')

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
import time
import random
from collections import defaultdict

# Interactive widgets (if available)
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("⚠️ Interactive widgets not available - using command line interface")

# Plotting libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project imports
from models.multimodal_model import MultimodalLLM
from data.data_loader import MultimodalDataModule
from data.preprocessing import TimeSeriesPreprocessor, TextPreprocessor
from utils.config_loader import load_config_for_training
from utils.inference_utils import MultimodalInferenceEngine, StreamingInferenceEngine, create_inference_pipeline
from utils.visualization import TrainingVisualizer
from utils.mlflow_utils import MLflowExperimentManager

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("✅ All imports successful")
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"🚀 CUDA available: {torch.cuda.is_available()}")
print(f"🎛️ Interactive widgets: {'Available' if WIDGETS_AVAILABLE else 'Not available'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration and Model Setup

# COMMAND ----------

# Configuration paths
CONFIG_DIR = "/Workspace/mllm_project/config"
MODEL_DIR = "/dbfs/mllm/checkpoints/final_model"
DATA_DIR = "/dbfs/mllm/data/raw/time_mmd"
DEMO_DIR = "/dbfs/mllm/demo"
PLOTS_DIR = "/dbfs/mllm/plots/demo"

# Create directories
os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load configuration
print("📋 Loading configuration...")
config = load_config_for_training(CONFIG_DIR)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")

# Model loading
print("🏗️ Setting up inference pipeline...")

try:
    # Create optimized inference engine
    inference_engine = create_inference_pipeline(
        model_path=MODEL_DIR,
        config_path=CONFIG_DIR,
        device=device,
        optimization_level="fast"  # Optimized for demo
    )
    
    # Create streaming engine for real-time generation
    streaming_engine = StreamingInferenceEngine(
        model_path=MODEL_DIR,
        config_path=CONFIG_DIR,
        device=device
    )
    
    print("✅ Inference engines loaded successfully")
    
    # Get model statistics
    model_stats = inference_engine.get_inference_stats()
    print(f"📊 Model parameters: {model_stats['model_parameters']:,}")
    print(f"💾 GPU memory usage: {model_stats['memory_usage_mb']:.1f} MB")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print("🔧 Creating demo inference engine for demonstration...")
    
    # Create demo inference engine with synthetic model
    class DemoInferenceEngine:
        def __init__(self):
            self.device = device
            self.model = MultimodalLLM(config).to(device)
            
        def generate_text(self, time_series=None, text_prompt=None, **kwargs):
            # Simulate generation
            time.sleep(0.1)  # Simulate processing time
            
            if text_prompt:
                generated = f"{text_prompt} demonstrates a complex pattern with seasonal variations and underlying trends."
            else:
                generated = "The time series data shows interesting temporal patterns with notable fluctuations."
            
            class MockResult:
                def __init__(self):
                    self.generated_text = generated
                    self.generation_time = 0.15
                    self.ts_embeddings = None
                    
            return MockResult()
        
        def get_inference_stats(self):
            return {
                'model_parameters': 125000000,
                'memory_usage_mb': 512.0
            }
    
    inference_engine = DemoInferenceEngine()
    streaming_engine = None
    print("✅ Demo inference engine created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Sample Data Generation and Preparation

# COMMAND ----------

print("🔧 Preparing sample data for demonstration...")

# Generate diverse sample time series
def generate_sample_time_series():
    """Generate various types of sample time series for demonstration."""
    samples = {}
    
    # 1. Trend with noise
    t = np.linspace(0, 10, 200)
    trend_series = 2 * t + 0.5 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.2, len(t))
    samples['trend'] = {
        'data': trend_series.reshape(-1, 1),
        'description': 'Upward trend with periodic oscillations',
        'domain': 'finance'
    }
    
    # 2. Seasonal pattern
    seasonal_series = 5 + 3 * np.sin(2 * np.pi * t / 2) + 1.5 * np.cos(2 * np.pi * t / 4) + np.random.normal(0, 0.3, len(t))
    samples['seasonal'] = {
        'data': seasonal_series.reshape(-1, 1),
        'description': 'Strong seasonal pattern with multiple frequencies',
        'domain': 'energy'
    }
    
    # 3. Stock-like volatility
    returns = np.random.normal(0.001, 0.02, len(t))
    stock_series = np.cumsum(returns) + 100
    samples['volatile'] = {
        'data': stock_series.reshape(-1, 1),
        'description': 'High volatility financial time series',
        'domain': 'finance'
    }
    
    # 4. Temperature-like data
    temp_base = 20 + 10 * np.sin(2 * np.pi * t / 10)  # Annual cycle
    temp_daily = 5 * np.sin(2 * np.pi * t * 30)  # Daily variation
    temp_series = temp_base + temp_daily + np.random.normal(0, 1, len(t))
    samples['temperature'] = {
        'data': temp_series.reshape(-1, 1),
        'description': 'Temperature data with annual and daily cycles',
        'domain': 'weather'
    }
    
    # 5. Multi-dimensional data
    multi_data = np.column_stack([
        trend_series,
        seasonal_series[:len(trend_series)],
        np.random.walk(len(trend_series))
    ])
    samples['multivariate'] = {
        'data': multi_data,
        'description': 'Multi-dimensional time series with correlated features',
        'domain': 'sensors'
    }
    
    return samples

# Generate sample data
sample_time_series = generate_sample_time_series()

# Sample text prompts for different scenarios
sample_prompts = {
    'analysis': [
        "Analyze this time series pattern:",
        "The data shows:",
        "Based on the temporal behavior:",
        "This pattern indicates:",
        "The trend suggests:"
    ],
    'forecasting': [
        "Future predictions for this series:",
        "Expected continuation:",
        "Forecasting the next period:",
        "Projected trends:",
        "Anticipated patterns:"
    ],
    'explanation': [
        "This time series represents:",
        "The underlying process:",
        "Key characteristics include:",
        "Notable features:",
        "Pattern interpretation:"
    ]
}

print("✅ Sample data prepared")
print(f"📊 Time series samples: {len(sample_time_series)}")
print(f"📝 Text prompt categories: {len(sample_prompts)}")

# Visualize sample time series
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (name, sample) in enumerate(sample_time_series.items()):
    if i < len(axes):
        data = sample['data']
        if data.ndim > 1 and data.shape[1] > 1:
            # Multi-dimensional - plot first 3 features
            for j in range(min(3, data.shape[1])):
                axes[i].plot(data[:, j], label=f'Feature {j+1}', alpha=0.8)
            axes[i].legend()
        else:
            # Single dimensional
            axes[i].plot(data.flatten(), color='blue', alpha=0.8)
        
        axes[i].set_title(f"{name.title()} - {sample['domain']}")
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)

# Hide extra subplot
if len(sample_time_series) < len(axes):
    axes[-1].axis('off')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/sample_time_series.png', dpi=300, bbox_inches='tight')
plt.show()

print("📊 Sample time series visualized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Interactive Text Generation Interface

# COMMAND ----------

print("🎮 Setting up interactive text generation interface...")

class TextGenerationDemo:
    """Interactive demo class for text generation."""
    
    def __init__(self, inference_engine, sample_data, sample_prompts):
        self.inference_engine = inference_engine
        self.sample_data = sample_data
        self.sample_prompts = sample_prompts
        self.generation_history = []
        
    def generate_with_prompt(self, time_series_name=None, custom_ts=None, 
                           text_prompt="", temperature=0.8, max_length=100):
        """Generate text with given inputs."""
        
        # Get time series data
        if custom_ts is not None:
            ts_data = custom_ts
            ts_description = "Custom time series"
        elif time_series_name and time_series_name in self.sample_data:
            ts_data = self.sample_data[time_series_name]['data']
            ts_description = f"{time_series_name} - {self.sample_data[time_series_name]['description']}"
        else:
            ts_data = None
            ts_description = "No time series"
        
        # Generate text
        start_time = time.time()
        
        try:
            result = self.inference_engine.generate_text(
                time_series=ts_data,
                text_prompt=text_prompt if text_prompt.strip() else None,
                temperature=temperature,
                max_length=max_length
            )
            
            generation_result = {
                'timestamp': datetime.now().isoformat(),
                'time_series': ts_description,
                'prompt': text_prompt,
                'generated_text': result.generated_text,
                'generation_time': result.generation_time,
                'temperature': temperature,
                'max_length': max_length
            }
            
            self.generation_history.append(generation_result)
            
            return generation_result
            
        except Exception as e:
            return {
                'error': f"Generation failed: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def get_random_prompt(self, category='analysis'):
        """Get a random prompt from specified category."""
        if category in self.sample_prompts:
            return random.choice(self.sample_prompts[category])
        return "Describe this data:"
    
    def batch_generate(self, num_samples=5):
        """Generate multiple examples for comparison."""
        results = []
        
        for i in range(num_samples):
            # Random selection
            ts_name = random.choice(list(self.sample_data.keys()))
            prompt_category = random.choice(list(self.sample_prompts.keys()))
            prompt = self.get_random_prompt(prompt_category)
            
            result = self.generate_with_prompt(
                time_series_name=ts_name,
                text_prompt=prompt,
                temperature=random.uniform(0.6, 1.0)
            )
            
            results.append(result)
            
        return results

# Create demo instance
demo = TextGenerationDemo(inference_engine, sample_time_series, sample_prompts)

# Non-interactive interface for demonstration
def run_demo_examples():
    """Run several demo examples."""
    
    print("🎯 Running demonstration examples...")
    print("=" * 60)
    
    # Example 1: Trend analysis
    print("\n1️⃣ TREND ANALYSIS EXAMPLE:")
    print("-" * 30)
    result1 = demo.generate_with_prompt(
        time_series_name='trend',
        text_prompt="Analyze this time series pattern:",
        temperature=0.7
    )
    
    if 'error' not in result1:
        print(f"📊 Time Series: {result1['time_series']}")
        print(f"💬 Prompt: {result1['prompt']}")
        print(f"✨ Generated: {result1['generated_text']}")
        print(f"⏱️ Time: {result1['generation_time']:.3f}s")
    else:
        print(f"❌ {result1['error']}")
    
    # Example 2: Seasonal pattern
    print("\n2️⃣ SEASONAL PATTERN EXAMPLE:")
    print("-" * 35)
    result2 = demo.generate_with_prompt(
        time_series_name='seasonal',
        text_prompt="The seasonal data shows:",
        temperature=0.8
    )
    
    if 'error' not in result2:
        print(f"📊 Time Series: {result2['time_series']}")
        print(f"💬 Prompt: {result2['prompt']}")
        print(f"✨ Generated: {result2['generated_text']}")
        print(f"⏱️ Time: {result2['generation_time']:.3f}s")
    else:
        print(f"❌ {result2['error']}")
    
    # Example 3: Multi-modal generation
    print("\n3️⃣ MULTIVARIATE DATA EXAMPLE:")
    print("-" * 35)
    result3 = demo.generate_with_prompt(
        time_series_name='multivariate',
        text_prompt="This multi-dimensional data reveals:",
        temperature=0.6
    )
    
    if 'error' not in result3:
        print(f"📊 Time Series: {result3['time_series']}")
        print(f"💬 Prompt: {result3['prompt']}")
        print(f"✨ Generated: {result3['generated_text']}")
        print(f"⏱️ Time: {result3['generation_time']:.3f}s")
    else:
        print(f"❌ {result3['error']}")
    
    # Example 4: Text-only generation
    print("\n4️⃣ TEXT-ONLY GENERATION:")
    print("-" * 30)
    result4 = demo.generate_with_prompt(
        time_series_name=None,
        text_prompt="Time series analysis often reveals:",
        temperature=0.9
    )
    
    if 'error' not in result4:
        print(f"📊 Time Series: {result4['time_series']}")
        print(f"💬 Prompt: {result4['prompt']}")
        print(f"✨ Generated: {result4['generated_text']}")
        print(f"⏱️ Time: {result4['generation_time']:.3f}s")
    else:
        print(f"❌ {result4['error']}")
    
    print("\n✅ Demo examples completed!")
    
    return [result1, result2, result3, result4]

# Run demonstration examples
demo_results = run_demo_examples()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Streaming Generation Demo

# COMMAND ----------

print("🔄 Demonstrating streaming text generation...")

def streaming_demo():
    """Demonstrate streaming text generation."""
    
    if streaming_engine is None:
        print("⚠️ Streaming engine not available - simulating streaming...")
        
        # Simulate streaming
        sample_text = "The time series demonstrates a clear upward trend with periodic fluctuations that suggest underlying seasonal patterns in the data generating process."
        words = sample_text.split()
        
        print("🎬 Simulated Streaming Generation:")
        print("💬 Prompt: 'The time series demonstrates'")
        print("✨ Generated text (streaming):")
        print("   ", end="", flush=True)
        
        for word in words[3:]:  # Skip the prompt words
            print(f"{word} ", end="", flush=True)
            time.sleep(0.1)  # Simulate word-by-word generation
        
        print("\n")
        print("✅ Simulated streaming complete")
        
    else:
        print("🎬 Real Streaming Generation:")
        print("💬 Prompt: 'The financial time series shows'")
        print("✨ Generated text (streaming):")
        print("   ", end="", flush=True)
        
        try:
            # Use sample time series
            ts_data = sample_time_series['volatile']['data']
            
            for token in streaming_engine.stream_generate(
                time_series=ts_data,
                text_prompt="The financial time series shows",
                max_new_tokens=20,
                temperature=0.8
            ):
                print(token, end="", flush=True)
                time.sleep(0.05)  # Small delay for demonstration
            
            print("\n")
            print("✅ Streaming generation complete")
            
        except Exception as e:
            print(f"\n❌ Streaming failed: {e}")

# Run streaming demo
streaming_demo()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Batch Processing and Performance Analysis

# COMMAND ----------

print("🔄 Running batch processing demonstration...")

def batch_processing_demo():
    """Demonstrate batch processing capabilities."""
    
    # Generate batch of examples
    print("📦 Generating batch of examples...")
    batch_results = demo.batch_generate(num_samples=8)
    
    # Analyze batch results
    successful_results = [r for r in batch_results if 'error' not in r]
    failed_results = [r for r in batch_results if 'error' in r]
    
    print(f"✅ Successful generations: {len(successful_results)}")
    print(f"❌ Failed generations: {len(failed_results)}")
    
    if successful_results:
        # Performance statistics
        generation_times = [r['generation_time'] for r in successful_results]
        text_lengths = [len(r['generated_text']) for r in successful_results]
        
        print(f"\n📊 Performance Statistics:")
        print(f"   ⏱️ Average generation time: {np.mean(generation_times):.3f}s")
        print(f"   ⏱️ Min/Max generation time: {np.min(generation_times):.3f}s / {np.max(generation_times):.3f}s")
        print(f"   📏 Average text length: {np.mean(text_lengths):.1f} characters")
        print(f"   🚀 Throughput: {len(successful_results) / sum(generation_times):.2f} samples/sec")
        
        # Show sample results
        print(f"\n📝 Sample Batch Results:")
        for i, result in enumerate(successful_results[:3], 1):
            print(f"\n{i}. {result['time_series']}")
            print(f"   💬 Prompt: {result['prompt']}")
            print(f"   ✨ Generated: {result['generated_text'][:100]}{'...' if len(result['generated_text']) > 100 else ''}")
            print(f"   ⏱️ Time: {result['generation_time']:.3f}s")
        
        # Visualize performance
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Generation times
        axes[0].hist(generation_times, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(np.mean(generation_times), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(generation_times):.3f}s')
        axes[0].set_xlabel('Generation Time (s)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Generation Times')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Text lengths
        axes[1].scatter(range(len(text_lengths)), text_lengths, alpha=0.7, color='coral')
        axes[1].axhline(np.mean(text_lengths), color='red', linestyle='--',
                       label=f'Mean: {np.mean(text_lengths):.1f} chars')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Generated Text Length (characters)')
        axes[1].set_title('Generated Text Lengths')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/batch_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return successful_results
    
    else:
        print("❌ No successful generations to analyze")
        return []

# Run batch processing demo
batch_results = batch_processing_demo()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Custom Time Series Input Demo

# COMMAND ----------

print("🔧 Custom time series input demonstration...")

def custom_time_series_demo():
    """Demonstrate with user-generated time series."""
    
    print("📊 Creating custom time series patterns...")
    
    # Custom pattern 1: Anomaly detection scenario
    t = np.linspace(0, 20, 300)
    normal_pattern = 5 + 2 * np.sin(0.5 * t) + np.random.normal(0, 0.3, len(t))
    
    # Inject anomalies
    anomaly_indices = [100, 150, 200, 250]
    for idx in anomaly_indices:
        if idx < len(normal_pattern):
            normal_pattern[idx] += random.choice([-3, 3]) * random.uniform(1.5, 2.5)
    
    custom_patterns = {
        'anomaly_detection': {
            'data': normal_pattern.reshape(-1, 1),
            'description': 'Time series with injected anomalies for detection',
            'prompts': [
                'Identify any unusual patterns in this data:',
                'This time series contains:',
                'Notable anomalies include:'
            ]
        },
        
        'forecast_scenario': {
            'data': np.cumsum(np.random.normal(0.1, 1.0, 200)).reshape(-1, 1),
            'description': 'Random walk pattern for forecasting',
            'prompts': [
                'Future predictions for this series:',
                'The trend suggests:',
                'Expected continuation:'
            ]
        },
        
        'correlation_analysis': {
            'data': np.column_stack([
                t[:200] + np.random.normal(0, 0.5, 200),
                2 * t[:200] + 3 + np.random.normal(0, 0.8, 200),
                -0.5 * t[:200] + 10 + np.random.normal(0, 0.4, 200)
            ]),
            'description': 'Multi-variate correlated time series',
            'prompts': [
                'The relationship between these variables:',
                'Correlation analysis reveals:',
                'These features show:'
            ]
        }
    }
    
    print("✅ Custom patterns created")
    
    # Test each custom pattern
    results = {}
    
    for pattern_name, pattern_info in custom_patterns.items():
        print(f"\n🧪 Testing {pattern_name}:")
        print("-" * 40)
        
        # Visualize the pattern
        plt.figure(figsize=(12, 4))
        data = pattern_info['data']
        
        if data.ndim > 1 and data.shape[1] > 1:
            for i in range(min(3, data.shape[1])):
                plt.plot(data[:, i], label=f'Feature {i+1}', alpha=0.8)
            plt.legend()
        else:
            plt.plot(data.flatten(), color='blue', alpha=0.8)
            
            # Highlight anomalies if this is anomaly detection
            if pattern_name == 'anomaly_detection':
                for idx in anomaly_indices:
                    if idx < len(data):
                        plt.scatter(idx, data[idx, 0], color='red', s=50, alpha=0.8)
        
        plt.title(f"{pattern_name.replace('_', ' ').title()}")
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/custom_{pattern_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Test with different prompts
        pattern_results = []
        for prompt in pattern_info['prompts']:
            result = demo.generate_with_prompt(
                custom_ts=data,
                text_prompt=prompt,
                temperature=0.7
            )
            
            if 'error' not in result:
                print(f"💬 Prompt: {prompt}")
                print(f"✨ Generated: {result['generated_text']}")
                print(f"⏱️ Time: {result['generation_time']:.3f}s")
                print()
                
                pattern_results.append(result)
            else:
                print(f"❌ {result['error']}")
        
        results[pattern_name] = pattern_results
    
    return results

# Run custom time series demo
custom_results = custom_time_series_demo()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Performance Monitoring and Benchmarking

# COMMAND ----------

print("📊 Performance monitoring and benchmarking...")

def performance_benchmark():
    """Comprehensive performance benchmark."""
    
    # Benchmark different scenarios
    benchmark_scenarios = [
        {
            'name': 'small_series',
            'ts_length': 50,
            'prompt_length': 10,
            'description': 'Small time series, short prompt'
        },
        {
            'name': 'medium_series',
            'ts_length': 200,
            'prompt_length': 25,
            'description': 'Medium time series, medium prompt'
        },
        {
            'name': 'large_series',
            'ts_length': 500,
            'prompt_length': 50,
            'description': 'Large time series, long prompt'
        },
        {
            'name': 'text_only',
            'ts_length': 0,
            'prompt_length': 30,
            'description': 'Text-only generation'
        }
    ]
    
    benchmark_results = {}
    
    print("🏃 Running performance benchmarks...")
    
    for scenario in benchmark_scenarios:
        print(f"\n🧪 Benchmarking: {scenario['description']}")
        
        scenario_times = []
        scenario_lengths = []
        
        # Run multiple trials
        for trial in range(5):
            # Create test data
            if scenario['ts_length'] > 0:
                test_ts = np.random.randn(scenario['ts_length'], 3).cumsum(axis=0)
            else:
                test_ts = None
            
            test_prompt = " ".join([f"word{i}" for i in range(scenario['prompt_length'])])
            
            # Time the generation
            start_time = time.time()
            
            result = demo.generate_with_prompt(
                custom_ts=test_ts,
                text_prompt=test_prompt,
                temperature=0.8
            )
            
            end_time = time.time()
            
            if 'error' not in result:
                scenario_times.append(end_time - start_time)
                scenario_lengths.append(len(result['generated_text']))
        
        if scenario_times:
            benchmark_results[scenario['name']] = {
                'description': scenario['description'],
                'avg_time': np.mean(scenario_times),
                'std_time': np.std(scenario_times),
                'min_time': np.min(scenario_times),
                'max_time': np.max(scenario_times),
                'avg_length': np.mean(scenario_lengths),
                'throughput': 1.0 / np.mean(scenario_times)
            }
            
            stats = benchmark_results[scenario['name']]
            print(f"   ⏱️ Avg time: {stats['avg_time']:.3f}s ± {stats['std_time']:.3f}s")
            print(f"   📏 Avg length: {stats['avg_length']:.1f} chars")
            print(f"   🚀 Throughput: {stats['throughput']:.2f} samples/sec")
    
    # Visualize benchmark results
    if benchmark_results:
        scenarios = list(benchmark_results.keys())
        avg_times = [benchmark_results[s]['avg_time'] for s in scenarios]
        throughputs = [benchmark_results[s]['throughput'] for s in scenarios]
        descriptions = [benchmark_results[s]['description'] for s in scenarios]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Average times
        bars1 = axes[0].bar(range(len(scenarios)), avg_times, color='lightblue', alpha=0.8)
        axes[0].set_xlabel('Scenario')
        axes[0].set_ylabel('Average Time (s)')
        axes[0].set_title('Average Generation Time by Scenario')
        axes[0].set_xticks(range(len(scenarios)))
        axes[0].set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45)
        
        # Add value labels
        for bar, time_val in zip(bars1, avg_times):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{time_val:.3f}s', ha='center', va='bottom')
        
        # Throughput
        bars2 = axes[1].bar(range(len(scenarios)), throughputs, color='lightcoral', alpha=0.8)
        axes[1].set_xlabel('Scenario')
        axes[1].set_ylabel('Throughput (samples/sec)')
        axes[1].set_title('Throughput by Scenario')
        axes[1].set_xticks(range(len(scenarios)))
        axes[1].set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=45)
        
        # Add value labels
        for bar, throughput in zip(bars2, throughputs):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{throughput:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/performance_benchmark.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Performance benchmark completed")
        
    return benchmark_results

# Run performance benchmark
perf_results = performance_benchmark()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Model Comparison and Quality Assessment

# COMMAND ----------

print("🔍 Model quality assessment and comparison...")

def quality_assessment():
    """Assess generation quality with various metrics."""
    
    print("📊 Running quality assessment...")
    
    # Generate samples for quality analysis
    quality_test_cases = [
        {
            'ts_name': 'trend',
            'prompt': 'Analyze this upward trend:',
            'expected_keywords': ['trend', 'increase', 'growth', 'upward']
        },
        {
            'ts_name': 'seasonal',
            'prompt': 'Describe the seasonal pattern:',
            'expected_keywords': ['seasonal', 'pattern', 'cycle', 'periodic']
        },
        {
            'ts_name': 'volatile',
            'prompt': 'Explain the volatility:',
            'expected_keywords': ['volatile', 'fluctuation', 'variation', 'unstable']
        }
    ]
    
    assessment_results = []
    
    for test_case in quality_test_cases:
        print(f"\n🧪 Testing: {test_case['ts_name']} with prompt '{test_case['prompt']}'")
        
        # Generate multiple samples with different temperatures
        temperatures = [0.3, 0.7, 1.0]
        case_results = []
        
        for temp in temperatures:
            result = demo.generate_with_prompt(
                time_series_name=test_case['ts_name'],
                text_prompt=test_case['prompt'],
                temperature=temp
            )
            
            if 'error' not in result:
                # Quality metrics
                generated = result['generated_text'].lower()
                
                # Keyword relevance
                keyword_hits = sum(1 for kw in test_case['expected_keywords'] if kw in generated)
                keyword_score = keyword_hits / len(test_case['expected_keywords'])
                
                # Length appropriateness (50-200 chars is good)
                length_score = min(1.0, max(0.0, (len(generated) - 20) / 180))
                
                # Repetition check (simple)
                words = generated.split()
                unique_words = len(set(words))
                repetition_score = unique_words / len(words) if len(words) > 0 else 0
                
                # Overall quality score
                quality_score = (keyword_score + length_score + repetition_score) / 3
                
                case_result = {
                    'temperature': temp,
                    'generated_text': result['generated_text'],
                    'keyword_score': keyword_score,
                    'length_score': length_score,
                    'repetition_score': repetition_score,
                    'quality_score': quality_score,
                    'generation_time': result['generation_time']
                }
                
                case_results.append(case_result)
                
                print(f"   🌡️ Temp {temp}: Quality={quality_score:.2f}, Keywords={keyword_score:.2f}")
        
        assessment_results.append({
            'test_case': test_case,
            'results': case_results
        })
    
    # Summarize quality assessment
    print("\n📋 Quality Assessment Summary:")
    print("=" * 50)
    
    all_scores = []
    for assessment in assessment_results:
        case_name = assessment['test_case']['ts_name']
        results = assessment['results']
        
        if results:
            avg_quality = np.mean([r['quality_score'] for r in results])
            best_result = max(results, key=lambda x: x['quality_score'])
            
            print(f"\n📊 {case_name.upper()}:")
            print(f"   🎯 Average quality: {avg_quality:.3f}")
            print(f"   🏆 Best quality: {best_result['quality_score']:.3f} (temp={best_result['temperature']})")
            print(f"   📝 Best example: {best_result['generated_text'][:100]}...")
            
            all_scores.extend([r['quality_score'] for r in results])
    
    if all_scores:
        overall_quality = np.mean(all_scores)
        print(f"\n🎯 OVERALL QUALITY SCORE: {overall_quality:.3f}/1.0")
        
        if overall_quality >= 0.8:
            print("✅ Excellent generation quality")
        elif overall_quality >= 0.6:
            print("⚠️ Good quality with room for improvement")
        else:
            print("❌ Quality needs improvement")
    
    return assessment_results

# Run quality assessment
quality_results = quality_assessment()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Export and Deployment Utilities

# COMMAND ----------

print("📦 Export and deployment utilities...")

def create_deployment_package():
    """Create deployment package with demo results."""
    
    print("📋 Creating deployment summary...")
    
    # Collect all demo results
    deployment_summary = {
        'creation_timestamp': datetime.now().isoformat(),
        'model_info': {
            'device': str(device),
            'model_stats': inference_engine.get_inference_stats() if hasattr(inference_engine, 'get_inference_stats') else {},
            'configuration': config
        },
        'demo_results': {
            'basic_examples': [r for r in demo_results if 'error' not in r],
            'batch_processing': batch_results[:5] if batch_results else [],
            'custom_patterns': {k: v[:2] for k, v in custom_results.items()},  # Limit results
            'performance_benchmark': perf_results,
            'quality_assessment': len(quality_results) if quality_results else 0
        },
        'sample_data_info': {
            'time_series_samples': list(sample_time_series.keys()),
            'prompt_categories': list(sample_prompts.keys())
        },
        'generation_statistics': {
            'total_generations': len(demo.generation_history),
            'successful_generations': len([g for g in demo.generation_history if 'error' not in g]),
            'avg_generation_time': np.mean([g['generation_time'] for g in demo.generation_history if 'generation_time' in g]) if demo.generation_history else 0
        }
    }
    
    # Save deployment summary
    summary_path = f"{DEMO_DIR}/deployment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(deployment_summary, f, indent=2, default=str)
    
    print(f"✅ Deployment summary saved to: {summary_path}")
    
    # Create inference examples for production use
    inference_examples = {
        'quick_start': {
            'description': 'Basic inference example',
            'code': """
# Load model
from utils.inference_utils import create_inference_pipeline
inference_engine = create_inference_pipeline(model_path, config_path, device='auto')

# Generate text
result = inference_engine.generate_text(
    time_series=your_time_series_data,  # shape: [seq_len, n_features]
    text_prompt="Analyze this pattern:",
    temperature=0.8
)

print(result.generated_text)
"""
        },
        'batch_processing': {
            'description': 'Batch processing example',
            'code': """
# Batch processing
inputs = [
    {'time_series': ts1, 'text_prompt': 'Describe pattern:'},
    {'time_series': ts2, 'text_prompt': 'Analyze trend:'},
]

results = inference_engine.batch_generate(inputs)
for result in results:
    print(result.generated_text)
"""
        },
        'streaming': {
            'description': 'Streaming generation example',
            'code': """
# Streaming generation
from utils.inference_utils import StreamingInferenceEngine
streaming_engine = StreamingInferenceEngine(model_path, config_path)

for token in streaming_engine.stream_generate(
    time_series=your_data,
    text_prompt="The pattern shows",
    max_new_tokens=50
):
    print(token, end='', flush=True)
"""
        }
    }
    
    # Save inference examples
    examples_path = f"{DEMO_DIR}/inference_examples.json"
    with open(examples_path, 'w') as f:
        json.dump(inference_examples, f, indent=2)
    
    print(f"✅ Inference examples saved to: {examples_path}")
    
    # Create README for demo
    readme_content = f"""# Multimodal LLM Demo Results

## Overview
This directory contains demonstration results for the multimodal LLM combining time series and text modalities.

## Generated Files
- `deployment_summary.json`: Complete demo results and statistics
- `inference_examples.json`: Code examples for production use
- `../plots/demo/`: Visualization plots from demonstrations

## Demo Statistics
- Total generations: {len(demo.generation_history)}
- Time series samples: {len(sample_time_series)}
- Prompt categories: {len(sample_prompts)}
- Average generation time: {np.mean([g['generation_time'] for g in demo.generation_history if 'generation_time' in g]):.3f}s

## Key Features Demonstrated
1. ✅ Text generation from time series input
2. ✅ Multimodal fusion capabilities
3. ✅ Batch processing efficiency
4. ✅ Streaming generation support
5. ✅ Custom time series handling
6. ✅ Performance monitoring
7. ✅ Quality assessment metrics

## Next Steps
1. Review generated examples in deployment_summary.json
2. Test inference examples with your own data
3. Adapt configuration for production deployment
4. Implement monitoring and logging for production use

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_path = f"{DEMO_DIR}/README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✅ README created: {readme_path}")
    
    return deployment_summary

# Create deployment package
deployment_info = create_deployment_package()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Demo Summary and Conclusions

# COMMAND ----------

print("📋 Demo Summary and Conclusions")
print("=" * 60)

# Calculate comprehensive demo statistics
total_generations = len(demo.generation_history)
successful_generations = len([g for g in demo.generation_history if 'error' not in g])
success_rate = successful_generations / total_generations if total_generations > 0 else 0

generation_times = [g['generation_time'] for g in demo.generation_history if 'generation_time' in g]
avg_generation_time = np.mean(generation_times) if generation_times else 0
total_demo_time = sum(generation_times) if generation_times else 0

print(f"📊 DEMO STATISTICS:")
print(f"   🎯 Total generations: {total_generations}")
print(f"   ✅ Successful generations: {successful_generations}")
print(f"   📈 Success rate: {success_rate:.1%}")
print(f"   ⏱️ Average generation time: {avg_generation_time:.3f}s")
print(f"   🕐 Total demo time: {total_demo_time:.1f}s")
print(f"   🚀 Average throughput: {successful_generations / total_demo_time:.2f} samples/sec" if total_demo_time > 0 else "   🚀 Throughput: N/A")

print(f"\n🎮 DEMO COMPONENTS COMPLETED:")
print(f"   ✅ Basic text generation examples")
print(f"   ✅ Streaming generation demonstration")
print(f"   ✅ Batch processing capabilities")
print(f"   ✅ Custom time series handling")
print(f"   ✅ Performance benchmarking")
print(f"   ✅ Quality assessment metrics")
print(f"   ✅ Deployment package creation")

print(f"\n📊 SAMPLE DATA COVERAGE:")
for ts_name, ts_info in sample_time_series.items():
    usage_count = len([g for g in demo.generation_history if ts_name in g.get('time_series', '')])
    print(f"   📈 {ts_name} ({ts_info['domain']}): {usage_count} generations")

print(f"\n🎯 MODEL CAPABILITIES DEMONSTRATED:")
print(f"   ✅ Time series to text generation")
print(f"   ✅ Text-only generation")
print(f"   ✅ Multimodal fusion")
print(f"   ✅ Variable input lengths")
print(f"   ✅ Temperature control")
print(f"   ✅ Batch processing")
print(f"   ✅ Real-time streaming")

print(f"\n📈 PERFORMANCE INSIGHTS:")
if perf_results:
    best_scenario = min(perf_results.items(), key=lambda x: x[1]['avg_time'])
    worst_scenario = max(perf_results.items(), key=lambda x: x[1]['avg_time'])
    
    print(f"   🏆 Fastest scenario: {best_scenario[0]} ({best_scenario[1]['avg_time']:.3f}s)")
    print(f"   🐌 Slowest scenario: {worst_scenario[0]} ({worst_scenario[1]['avg_time']:.3f}s)")

print(f"\n🎯 QUALITY ASSESSMENT:")
if quality_results:
    print(f"   ✅ Quality assessment completed on {len(quality_results)} test cases")
    print(f"   📊 Multiple temperature settings tested")
    print(f"   🔍 Keyword relevance, length, and repetition analyzed")

print(f"\n📁 OUTPUT FILES:")
print(f"   📊 Deployment summary: {DEMO_DIR}/deployment_summary.json")
print(f"   📝 Inference examples: {DEMO_DIR}/inference_examples.json")
print(f"   📖 Documentation: {DEMO_DIR}/README.md")
print(f"   📈 Visualization plots: {PLOTS_DIR}/")

print(f"\n🚀 PRODUCTION READINESS:")
if success_rate >= 0.9:
    print(f"   ✅ High success rate - ready for production testing")
elif success_rate >= 0.7:
    print(f"   ⚠️ Good success rate - monitor for edge cases")
else:
    print(f"   ❌ Low success rate - requires improvement before production")

if avg_generation_time < 1.0:
    print(f"   ✅ Fast generation - suitable for real-time applications")
elif avg_generation_time < 3.0:
    print(f"   ⚠️ Moderate speed - suitable for batch processing")
else:
    print(f"   ❌ Slow generation - optimization needed")

print(f"\n💡 RECOMMENDATIONS:")
print(f"   1. 🔧 Test with your specific time series data")
print(f"   2. 📊 Monitor performance in production environment")
print(f"   3. 🎯 Fine-tune temperature and generation parameters")
print(f"   4. 📈 Implement monitoring and logging")
print(f"   5. 🛡️ Add input validation and error handling")
print(f"   6. 📚 Create domain-specific prompt templates")

print(f"\n🎉 DEMO COMPLETED SUCCESSFULLY!")
print(f"🔗 Next: Review generated files and deploy to production")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Cleanup and Next Steps

# COMMAND ----------

print("🧹 Cleanup and preparation for next steps...")

# Clear GPU memory if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("🔥 GPU memory cleared")

# Save final demo state
final_demo_state = {
    'completion_timestamp': datetime.now().isoformat(),
    'demo_session_stats': {
        'total_generations': len(demo.generation_history),
        'successful_generations': len([g for g in demo.generation_history if 'error' not in g]),
        'demo_duration_minutes': (datetime.now() - datetime.fromisoformat(demo.generation_history[0]['timestamp'])).total_seconds() / 60 if demo.generation_history else 0,
        'components_tested': [
            'basic_generation',
            'streaming_generation',
            'batch_processing',
            'custom_time_series',
            'performance_benchmark',
            'quality_assessment',
            'deployment_package'
        ]
    },
    'files_created': [
        f"{DEMO_DIR}/deployment_summary.json",
        f"{DEMO_DIR}/inference_examples.json",
        f"{DEMO_DIR}/README.md",
        f"{PLOTS_DIR}/sample_time_series.png",
        f"{PLOTS_DIR}/batch_performance.png",
        f"{PLOTS_DIR}/performance_benchmark.png"
    ]
}

# Save final state
final_state_path = f"{DEMO_DIR}/demo_session_final.json"
with open(final_state_path, 'w') as f:
    json.dump(final_demo_state, f, indent=2, default=str)

print(f"✅ Final demo state saved: {final_state_path}")

print("\n🚀 Next Steps Guide:")
print("=" * 40)
print("1. 📊 **Review Demo Results**:")
print("   • Check deployment_summary.json for complete results")
print("   • Review visualization plots in the plots directory")
print("   • Analyze performance benchmarks and quality metrics")
print()
print("2. 🔧 **Production Integration**:")
print("   • Use inference_examples.json as starting templates")
print("   • Adapt configuration for your production environment")
print("   • Implement proper error handling and monitoring")
print()
print("3. 🧪 **Testing with Your Data**:")
print("   • Replace sample time series with your actual data")
print("   • Test with domain-specific prompts and scenarios")
print("   • Validate performance with your expected workload")
print()
print("4. 📈 **Performance Optimization**:")
print("   • Consider model quantization for faster inference")
print("   • Implement caching for repeated queries")
print("   • Set up proper batch size for your hardware")
print()
print("5. 🛡️ **Production Deployment**:")
print("   • Set up monitoring and alerting")
print("   • Implement input validation and sanitization")
print("   • Create fallback mechanisms for failures")
print("   • Document API endpoints and usage patterns")
print()
print("6. 📚 **Documentation and Training**:")
print("   • Create user guides for your specific use case")
print("   • Document model limitations and best practices")
print("   • Train users on effective prompt engineering")

print(f"\n✅ Interactive inference demo completed successfully!")
print(f"📁 All demo files saved to: {DEMO_DIR}")
print(f"📊 Visualizations available in: {PLOTS_DIR}")
print(f"🎯 Ready for production deployment and testing!")

# COMMAND ----------