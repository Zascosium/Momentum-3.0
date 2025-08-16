# Databricks notebook source
"""
# Data Exploration and Analysis for Time-MMD Dataset

This notebook provides comprehensive data exploration and analysis for the Time-MMD (Multi-Domain Multimodal Dataset).
It helps understand the dataset structure, quality, and characteristics before training the multimodal LLM.

## Notebook Overview
1. Dataset Loading and Structure Analysis
2. Time Series Data Exploration
3. Text Data Analysis
4. Multimodal Alignment Exploration
5. Data Quality Assessment
6. Statistical Analysis and Visualizations
7. Preprocessing Pipeline Validation

## Prerequisites
- Time-MMD dataset downloaded and available in DBFS
- Required libraries installed
- Configuration files properly set up
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup and Imports

# COMMAND ----------

import sys
import os
from pathlib import Path

# Add project source to path
sys.path.append('/Workspace/mllm_project/src')

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
import warnings
warnings.filterwarnings('ignore')

# Databricks specific
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

# Project imports
from data.dataset import TimeMmdDataset, create_data_loaders
from data.preprocessing import TimeSeriesPreprocessor, TextPreprocessor
from utils.config_loader import load_config_for_training
from utils.visualization import TrainingVisualizer

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("✅ All imports successful")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration and Data Paths

# COMMAND ----------

# Configuration paths
CONFIG_DIR = "/Workspace/mllm_project/config"
DATA_DIR = "/dbfs/mllm/data/raw/time_mmd"
CACHE_DIR = "/dbfs/mllm/data/cache"
PLOTS_DIR = "/dbfs/mllm/plots/exploration"

# Load configuration
config = load_config_for_training(CONFIG_DIR)

# Create directories
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"📁 Data directory: {DATA_DIR}")
print(f"📁 Cache directory: {CACHE_DIR}")
print(f"📁 Plots directory: {PLOTS_DIR}")

# Display configuration summary
print("\n📋 Configuration Summary:")
print(f"- Domains included: {config['domains']['included']}")
print(f"- Time series max length: {config['time_series']['max_length']}")
print(f"- Text max length: {config['text']['tokenizer']['max_length']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Dataset Loading and Basic Statistics

# COMMAND ----------

# Initialize dataset (without actual loading yet)
try:
    dataset = TimeMmdDataset(
        data_dir=DATA_DIR,
        config=config,
        split="train",
        cache_data=True
    )
    
    print(f"✅ Dataset initialized successfully")
    print(f"📊 Total samples: {len(dataset)}")
    
    # Get dataset statistics
    if len(dataset) > 0:
        sample_info = dataset.get_sample_info(0)
        print(f"📈 Sample info: {sample_info}")
    
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    print("Note: This is expected if Time-MMD data is not available")
    
    # Create synthetic data for demonstration
    print("🔧 Creating synthetic data for demonstration...")
    
    # Synthetic data parameters
    n_samples = 1000
    domains = ['weather', 'finance', 'energy']
    
    synthetic_data = []
    for i in range(n_samples):
        domain = np.random.choice(domains)
        ts_length = np.random.randint(100, 512)
        n_features = np.random.randint(1, 5)
        
        # Generate synthetic time series
        time_series = np.random.randn(ts_length, n_features).cumsum(axis=0)
        
        # Generate synthetic text
        text_templates = {
            'weather': f"Weather conditions show temperature of {np.random.randint(15, 35)}°C with humidity at {np.random.randint(40, 80)}%",
            'finance': f"Stock price movement shows trend with volatility of {np.random.uniform(0.1, 0.5):.2f}",
            'energy': f"Energy consumption pattern indicates demand of {np.random.randint(100, 500)}MW"
        }
        
        synthetic_data.append({
            'domain': domain,
            'time_series': time_series,
            'text': text_templates[domain],
            'ts_length': ts_length,
            'n_features': n_features
        })
    
    print(f"✅ Created {len(synthetic_data)} synthetic samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Time Series Data Analysis

# COMMAND ----------

# Analyze time series characteristics
if 'synthetic_data' in locals():
    data_to_analyze = synthetic_data
else:
    # Use real dataset samples
    data_to_analyze = []
    for i in range(min(100, len(dataset))):
        sample = dataset.get_sample_info(i)
        data_to_analyze.append(sample)

print("📊 Time Series Analysis")
print("=" * 50)

# Collect statistics
ts_lengths = [sample['ts_length'] if 'ts_length' in sample else len(sample.get('time_series', [])) for sample in data_to_analyze]
domains = [sample['domain'] for sample in data_to_analyze]
n_features_list = [sample.get('n_features', 1) for sample in data_to_analyze]

# Basic statistics
print(f"📈 Time Series Length Statistics:")
print(f"  - Mean: {np.mean(ts_lengths):.1f}")
print(f"  - Median: {np.median(ts_lengths):.1f}")
print(f"  - Min: {np.min(ts_lengths)}")
print(f"  - Max: {np.max(ts_lengths)}")
print(f"  - Std: {np.std(ts_lengths):.1f}")

print(f"\n🏷️ Domain Distribution:")
domain_counts = pd.Series(domains).value_counts()
for domain, count in domain_counts.items():
    print(f"  - {domain}: {count} samples ({count/len(domains)*100:.1f}%)")

print(f"\n🔢 Feature Count Distribution:")
feature_counts = pd.Series(n_features_list).value_counts().sort_index()
for n_feat, count in feature_counts.items():
    print(f"  - {n_feat} features: {count} samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Visualizations - Time Series Characteristics

# COMMAND ----------

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Time series length distribution
axes[0, 0].hist(ts_lengths, bins=30, alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Time Series Length')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Time Series Lengths')
axes[0, 0].axvline(np.mean(ts_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(ts_lengths):.1f}')
axes[0, 0].legend()

# 2. Domain distribution
domain_counts.plot(kind='bar', ax=axes[0, 1], color='skyblue')
axes[0, 1].set_xlabel('Domain')
axes[0, 1].set_ylabel('Number of Samples')
axes[0, 1].set_title('Sample Distribution by Domain')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. Feature count distribution
feature_counts.plot(kind='bar', ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_xlabel('Number of Features')
axes[1, 0].set_ylabel('Number of Samples')
axes[1, 0].set_title('Distribution of Feature Counts')

# 4. Length vs Features scatter plot
axes[1, 1].scatter(ts_lengths, n_features_list, alpha=0.6, c=[hash(d) % 10 for d in domains])
axes[1, 1].set_xlabel('Time Series Length')
axes[1, 1].set_ylabel('Number of Features')
axes[1, 1].set_title('Time Series Length vs Number of Features')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/time_series_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Time series analysis plots saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Sample Time Series Visualization

# COMMAND ----------

# Visualize sample time series from different domains
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Get samples from different domains
unique_domains = list(set(domains))[:3]

for i, domain in enumerate(unique_domains):
    # Find a sample from this domain
    domain_samples = [s for s in data_to_analyze if s['domain'] == domain]
    
    if domain_samples:
        sample = domain_samples[0]
        
        if 'synthetic_data' in locals():
            # Use synthetic data
            sample_data = [s for s in synthetic_data if s['domain'] == domain][0]
            ts_data = sample_data['time_series']
        else:
            # Use real data (would need to load actual sample)
            ts_data = np.random.randn(200, 2).cumsum(axis=0)  # Placeholder
        
        # Plot time series
        time_steps = range(len(ts_data))
        
        for feat_idx in range(min(ts_data.shape[1], 3)):  # Plot up to 3 features
            axes[i].plot(time_steps, ts_data[:, feat_idx], 
                        label=f'Feature {feat_idx+1}', linewidth=1.5, alpha=0.8)
        
        axes[i].set_title(f'Sample Time Series - {domain.title()} Domain')
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/sample_time_series.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Sample time series plots saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Text Data Analysis

# COMMAND ----------

print("📝 Text Data Analysis")
print("=" * 50)

# Analyze text characteristics
text_lengths = []
word_counts = []
texts = []

for sample in data_to_analyze:
    if 'text' in sample:
        text = sample['text']
        texts.append(text)
        text_lengths.append(len(text))
        word_counts.append(len(text.split()))

if texts:
    print(f"📊 Text Statistics:")
    print(f"  - Number of text samples: {len(texts)}")
    print(f"  - Mean character length: {np.mean(text_lengths):.1f}")
    print(f"  - Mean word count: {np.mean(word_counts):.1f}")
    print(f"  - Min character length: {np.min(text_lengths)}")
    print(f"  - Max character length: {np.max(text_lengths)}")
    
    # Sample texts
    print(f"\n📄 Sample Texts:")
    for i, text in enumerate(texts[:5]):
        print(f"  {i+1}. [{domains[i]}] {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # Text length distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Character length distribution
    axes[0].hist(text_lengths, bins=20, alpha=0.7, edgecolor='black', color='coral')
    axes[0].set_xlabel('Character Length')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Text Character Lengths')
    axes[0].axvline(np.mean(text_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(text_lengths):.1f}')
    axes[0].legend()
    
    # Word count distribution
    axes[1].hist(word_counts, bins=20, alpha=0.7, edgecolor='black', color='lightblue')
    axes[1].set_xlabel('Word Count')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Text Word Counts')
    axes[1].axvline(np.mean(word_counts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(word_counts):.1f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/text_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Text analysis plots saved")

else:
    print("❌ No text data found in samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Data Quality Assessment

# COMMAND ----------

print("🔍 Data Quality Assessment")
print("=" * 50)

# Quality metrics
quality_metrics = {
    'total_samples': len(data_to_analyze),
    'valid_samples': 0,
    'missing_time_series': 0,
    'missing_text': 0,
    'valid_domains': 0,
    'length_outliers': 0
}

# Define quality criteria
min_ts_length = config['quality_control']['min_time_series_length']
max_ts_length = config['quality_control']['max_time_series_length']
min_text_length = config['quality_control']['min_text_length']

valid_samples = []

for sample in data_to_analyze:
    is_valid = True
    
    # Check time series
    ts_length = sample.get('ts_length', 0)
    if ts_length == 0:
        quality_metrics['missing_time_series'] += 1
        is_valid = False
    elif ts_length < min_ts_length or ts_length > max_ts_length:
        quality_metrics['length_outliers'] += 1
        is_valid = False
    
    # Check text
    text = sample.get('text', '')
    if not text or len(text) < min_text_length:
        quality_metrics['missing_text'] += 1
        is_valid = False
    
    # Check domain
    domain = sample.get('domain', '')
    if domain in config['domains']['included']:
        quality_metrics['valid_domains'] += 1
    else:
        is_valid = False
    
    if is_valid:
        quality_metrics['valid_samples'] += 1
        valid_samples.append(sample)

# Calculate percentages
total = quality_metrics['total_samples']
print(f"📊 Quality Assessment Results:")
print(f"  - Total samples: {total}")
print(f"  - Valid samples: {quality_metrics['valid_samples']} ({quality_metrics['valid_samples']/total*100:.1f}%)")
print(f"  - Missing time series: {quality_metrics['missing_time_series']} ({quality_metrics['missing_time_series']/total*100:.1f}%)")
print(f"  - Missing text: {quality_metrics['missing_text']} ({quality_metrics['missing_text']/total*100:.1f}%)")
print(f"  - Length outliers: {quality_metrics['length_outliers']} ({quality_metrics['length_outliers']/total*100:.1f}%)")
print(f"  - Valid domains: {quality_metrics['valid_domains']} ({quality_metrics['valid_domains']/total*100:.1f}%)")

# Quality score
quality_score = quality_metrics['valid_samples'] / total * 100
print(f"\n🎯 Overall Data Quality Score: {quality_score:.1f}%")

if quality_score > 80:
    print("✅ Excellent data quality")
elif quality_score > 60:
    print("⚠️ Good data quality with some issues")
else:
    print("❌ Poor data quality - review required")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Preprocessing Pipeline Validation

# COMMAND ----------

print("🔧 Preprocessing Pipeline Validation")
print("=" * 50)

try:
    # Initialize preprocessors
    ts_preprocessor = TimeSeriesPreprocessor(config['time_series'])
    text_preprocessor = TextPreprocessor(config['text'])
    
    print("✅ Preprocessors initialized successfully")
    
    # Test preprocessing on sample data
    if 'synthetic_data' in locals() and len(synthetic_data) > 0:
        sample = synthetic_data[0]
        
        # Test time series preprocessing
        ts_data = sample['time_series']
        print(f"📈 Original TS shape: {ts_data.shape}")
        
        processed_ts = ts_preprocessor.process(ts_data, fit=True)
        print(f"📈 Processed TS shape: {processed_ts.shape}")
        
        # Test text preprocessing
        text = sample['text']
        print(f"📝 Original text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        tokenized = text_preprocessor.tokenize_text(text)
        print(f"📝 Tokenized shape: {tokenized['input_ids'].shape}")
        print(f"📝 Attention mask shape: {tokenized['attention_mask'].shape}")
        
        # Decode back to verify
        decoded = text_preprocessor.tokenizer.decode(tokenized['input_ids'], skip_special_tokens=True)
        print(f"📝 Decoded text: '{decoded[:50]}{'...' if len(decoded) > 50 else ''}'")
        
        print("✅ Preprocessing pipeline validation successful")
        
    else:
        print("⚠️ No sample data available for preprocessing validation")
        
except Exception as e:
    print(f"❌ Preprocessing validation failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Correlation and Relationship Analysis

# COMMAND ----------

print("🔗 Correlation and Relationship Analysis")
print("=" * 50)

# Analyze relationships between time series and text characteristics
if texts and len(ts_lengths) > 0:
    
    # Create correlation matrix
    analysis_df = pd.DataFrame({
        'ts_length': ts_lengths[:len(texts)],
        'text_length': text_lengths[:len(ts_lengths)],
        'word_count': word_counts[:len(ts_lengths)],
        'n_features': n_features_list[:len(texts)]
    })
    
    # Compute correlations
    correlation_matrix = analysis_df.corr()
    
    print("📊 Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix: Time Series vs Text Characteristics')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Domain-specific analysis
    domain_stats = pd.DataFrame(data_to_analyze[:len(texts)])
    domain_stats['ts_length'] = ts_lengths[:len(texts)]
    domain_stats['text_length'] = text_lengths[:len(ts_lengths)]
    
    print("\n📈 Domain-specific Statistics:")
    for domain in unique_domains:
        domain_data = domain_stats[domain_stats['domain'] == domain]
        if len(domain_data) > 0:
            print(f"\n{domain.upper()}:")
            print(f"  - Avg TS length: {domain_data['ts_length'].mean():.1f}")
            print(f"  - Avg text length: {domain_data['text_length'].mean():.1f}")
            print(f"  - Sample count: {len(domain_data)}")
    
    print("✅ Correlation analysis completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Memory and Performance Analysis

# COMMAND ----------

print("💾 Memory and Performance Analysis")
print("=" * 50)

# Estimate memory requirements
sample_ts_size = np.mean(ts_lengths) * np.mean(n_features_list) * 4  # 4 bytes per float32
sample_text_size = np.mean(text_lengths) * 1  # 1 byte per character (rough estimate)

total_samples = len(data_to_analyze)
estimated_memory_mb = (sample_ts_size + sample_text_size) * total_samples / (1024 * 1024)

print(f"📊 Memory Estimates:")
print(f"  - Average TS sample size: {sample_ts_size/1024:.1f} KB")
print(f"  - Average text sample size: {sample_text_size/1024:.1f} KB")
print(f"  - Total samples: {total_samples}")
print(f"  - Estimated dataset memory: {estimated_memory_mb:.1f} MB")

# Training batch memory estimate
batch_size = config['data_loading']['batch_size']
max_ts_length = config['time_series']['max_length']
max_text_length = config['text']['tokenizer']['max_length']

batch_memory_mb = (
    batch_size * max_ts_length * np.mean(n_features_list) * 4 +  # Time series
    batch_size * max_text_length * 4                             # Text tokens
) / (1024 * 1024)

print(f"\n🏋️ Training Memory Estimates:")
print(f"  - Batch size: {batch_size}")
print(f"  - Estimated batch memory: {batch_memory_mb:.1f} MB")
print(f"  - Recommended GPU memory: {batch_memory_mb * 10:.0f} MB")

# Performance recommendations
print(f"\n🚀 Performance Recommendations:")
if batch_memory_mb > 1000:  # > 1GB per batch
    print("  ⚠️ Large batch memory detected - consider reducing batch size")
elif batch_memory_mb < 100:  # < 100MB per batch
    print("  ✅ Batch size can potentially be increased for better GPU utilization")
else:
    print("  ✅ Batch size appears optimal")

if estimated_memory_mb > 10000:  # > 10GB dataset
    print("  ⚠️ Large dataset - consider data streaming or chunking")
else:
    print("  ✅ Dataset size manageable for in-memory processing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary and Recommendations

# COMMAND ----------

print("📋 Data Exploration Summary")
print("=" * 60)

# Generate comprehensive summary
summary = {
    'dataset_stats': {
        'total_samples': len(data_to_analyze),
        'valid_samples': quality_metrics['valid_samples'],
        'quality_score': quality_score,
        'unique_domains': len(unique_domains)
    },
    'time_series_stats': {
        'avg_length': np.mean(ts_lengths),
        'length_range': (np.min(ts_lengths), np.max(ts_lengths)),
        'avg_features': np.mean(n_features_list)
    },
    'text_stats': {
        'avg_char_length': np.mean(text_lengths) if text_lengths else 0,
        'avg_word_count': np.mean(word_counts) if word_counts else 0
    },
    'memory_estimates': {
        'dataset_mb': estimated_memory_mb,
        'batch_mb': batch_memory_mb
    }
}

print("📊 DATASET OVERVIEW:")
print(f"  • Total samples: {summary['dataset_stats']['total_samples']:,}")
print(f"  • Valid samples: {summary['dataset_stats']['valid_samples']:,} ({quality_score:.1f}%)")
print(f"  • Domains covered: {summary['dataset_stats']['unique_domains']}")

print(f"\n📈 TIME SERIES CHARACTERISTICS:")
print(f"  • Average length: {summary['time_series_stats']['avg_length']:.1f} time steps")
print(f"  • Length range: {summary['time_series_stats']['length_range'][0]} - {summary['time_series_stats']['length_range'][1]}")
print(f"  • Average features: {summary['time_series_stats']['avg_features']:.1f}")

print(f"\n📝 TEXT CHARACTERISTICS:")
print(f"  • Average character length: {summary['text_stats']['avg_char_length']:.1f}")
print(f"  • Average word count: {summary['text_stats']['avg_word_count']:.1f}")

print(f"\n💾 MEMORY REQUIREMENTS:")
print(f"  • Dataset memory: {summary['memory_estimates']['dataset_mb']:.1f} MB")
print(f"  • Batch memory: {summary['memory_estimates']['batch_mb']:.1f} MB")

# Recommendations
print(f"\n🎯 RECOMMENDATIONS:")

if quality_score > 80:
    print("  ✅ Data quality is excellent - proceed with training")
elif quality_score > 60:
    print("  ⚠️ Data quality is good but consider additional filtering")
else:
    print("  ❌ Data quality needs improvement before training")

if summary['time_series_stats']['avg_length'] < 50:
    print("  ⚠️ Time series are relatively short - consider longer sequences")

if summary['text_stats']['avg_char_length'] < 50:
    print("  ⚠️ Text samples are short - consider text augmentation")

if summary['memory_estimates']['batch_mb'] > 1000:
    print("  ⚠️ Consider reducing batch size or sequence lengths")

print(f"\n💼 TRAINING READINESS:")
ready_score = (quality_score + 
               min(100, summary['time_series_stats']['avg_length']) + 
               min(100, summary['text_stats']['avg_char_length'])) / 3

if ready_score > 80:
    print("  🟢 Dataset is ready for training")
elif ready_score > 60:
    print("  🟡 Dataset is mostly ready with minor issues")
else:
    print("  🔴 Dataset needs significant preparation")

# Save summary
summary_path = f"{PLOTS_DIR}/exploration_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n✅ Exploration complete! Summary saved to {summary_path}")
print(f"📁 All plots saved to {PLOTS_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Next Steps

# COMMAND ----------

print("🚀 Next Steps for Model Training")
print("=" * 50)

print("Based on the data exploration, here are the recommended next steps:")
print()
print("1. 📊 DATA PREPARATION:")
print("   • Review and clean any low-quality samples identified")
print("   • Consider data augmentation for underrepresented domains")
print("   • Validate preprocessing pipeline with larger sample")
print()
print("2. 🏗️ MODEL CONFIGURATION:")
print("   • Adjust sequence lengths based on data characteristics")
print("   • Configure batch size based on memory analysis")
print("   • Set up domain-specific training strategies if needed")
print()
print("3. 🎯 TRAINING SETUP:")
print("   • Configure MLflow experiment tracking")
print("   • Set up distributed training if using multiple GPUs")
print("   • Prepare validation and test splits")
print()
print("4. 📈 MONITORING:")
print("   • Set up training metrics monitoring")
print("   • Configure early stopping based on validation performance")
print("   • Plan evaluation metrics for multimodal alignment")
print()
print("📍 Ready to proceed to notebook 02_model_training.py!")

# COMMAND ----------