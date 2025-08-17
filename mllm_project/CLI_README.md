# Multimodal LLM Pipeline CLI

A comprehensive command-line interface for training and deploying multimodal language models that combine time series and text data.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Make the CLI executable:
```bash
chmod +x cli.py
```

## Quick Start

### 1. Data Exploration
Analyze your dataset and assess data quality:
```bash
python cli.py explore --data-dir ./data --output-dir ./results/exploration
```

### 2. Model Training
Train the multimodal model:
```bash
python cli.py train --config-dir ./config --epochs 10 --batch-size 32
```

### 3. Model Evaluation
Evaluate the trained model:
```bash
python cli.py evaluate --model-path ./checkpoints/best_model.pt --output-dir ./results/evaluation
```

### 4. Inference Demo
Run interactive inference demonstrations:
```bash
python cli.py demo --model-path ./checkpoints/best_model.pt --interactive
```

### 5. Full Pipeline
Run the complete end-to-end pipeline:
```bash
python cli.py pipeline --config ./config/pipeline_config.yaml
```

### 6. Model Serving
Start the REST API for model serving:
```bash
python cli.py serve --model-path ./checkpoints/best_model.pt --port 8080
```

## Commands

### `explore` - Data Exploration
Performs comprehensive data exploration and quality assessment.

**Options:**
- `--data-dir`: Path to data directory
- `--config-dir`: Configuration directory (default: `config`)
- `--output-dir`: Output directory for results
- `--sample-size`: Number of samples to analyze (default: 1000)
- `--generate-report`: Generate HTML report
- `--domains`: Specific domains to analyze

**Example:**
```bash
python cli.py explore \
    --data-dir /path/to/data \
    --output-dir ./exploration_results \
    --sample-size 5000 \
    --generate-report
```

### `train` - Model Training
Trains the multimodal LLM with MLflow tracking.

**Options:**
- `--config-dir`: Configuration directory
- `--data-dir`: Data directory
- `--checkpoint-dir`: Directory for model checkpoints
- `--experiment-name`: MLflow experiment name
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--distributed`: Enable distributed training
- `--mixed-precision`: Enable mixed precision training
- `--resume-from`: Resume from checkpoint

**Example:**
```bash
python cli.py train \
    --config-dir ./config \
    --epochs 20 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --mixed-precision
```

### `evaluate` - Model Evaluation
Performs comprehensive model evaluation with metrics and visualizations.

**Options:**
- `--model-path`: Path to trained model (required)
- `--config-dir`: Configuration directory
- `--data-dir`: Test data directory
- `--output-dir`: Output directory
- `--test-split`: Data split to evaluate on
- `--generate-plots`: Generate visualization plots
- `--save-predictions`: Save model predictions

**Example:**
```bash
python cli.py evaluate \
    --model-path ./checkpoints/best_model.pt \
    --output-dir ./evaluation_results \
    --generate-plots \
    --save-predictions
```

### `demo` - Inference Demo
Provides interactive demonstrations of model capabilities.

**Options:**
- `--model-path`: Path to trained model (required)
- `--demo-dir`: Directory for demo outputs
- `--interactive`: Run in interactive mode
- `--batch-demo`: Run batch processing demo
- `--streaming`: Enable streaming generation
- `--num-examples`: Number of examples to generate
- `--temperature`: Generation temperature

**Example:**
```bash
# Interactive mode
python cli.py demo --model-path ./model.pt --interactive --streaming

# Batch demo
python cli.py demo --model-path ./model.pt --batch-demo --num-examples 20
```

### `pipeline` - Full Pipeline
Orchestrates the complete end-to-end pipeline.

**Options:**
- `--config`: Pipeline configuration file (required)
- `--stages`: Specific stages to run
- `--skip-stages`: Stages to skip
- `--output-dir`: Output directory
- `--resume`: Resume from previous run
- `--dry-run`: Show plan without executing

**Example:**
```bash
# Run full pipeline
python cli.py pipeline --config ./config/pipeline_config.yaml

# Run specific stages
python cli.py pipeline \
    --config ./config/pipeline_config.yaml \
    --stages train evaluate

# Resume from previous run
python cli.py pipeline \
    --config ./config/pipeline_config.yaml \
    --resume
```

### `serve` - Model Serving API
Starts a REST API server for model inference.

**Options:**
- `--model-path`: Path to trained model (required)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to listen on (default: 8080)
- `--workers`: Number of worker processes
- `--reload`: Enable auto-reload
- `--enable-cors`: Enable CORS support
- `--api-key`: API key for authentication

**Example:**
```bash
# Basic serving
python cli.py serve --model-path ./model.pt --port 8080

# With authentication
python cli.py serve \
    --model-path ./model.pt \
    --port 8080 \
    --api-key your-secret-key \
    --enable-cors
```

## Configuration

### Pipeline Configuration
The pipeline can be configured using YAML files. See `config/pipeline_config.yaml` for an example.

Key configuration sections:
- `explore`: Data exploration settings
- `train`: Training parameters
- `evaluate`: Evaluation settings
- `demo`: Demo configuration
- `global`: Global settings (device, logging, etc.)

### Model Configuration
Model architecture and training parameters are defined in:
- `config/model_config.yaml`: Model architecture
- `config/training_config.yaml`: Training parameters
- `config/data_config.yaml`: Data processing settings

## Example Scripts

The `examples/` directory contains ready-to-use scripts:

1. **run_exploration.sh**: Run data exploration
2. **run_training.sh**: Train the model
3. **run_full_pipeline.sh**: Execute complete pipeline
4. **run_inference_demo.sh**: Run inference demonstrations
5. **run_model_serving.sh**: Start API server
6. **example_pipeline.py**: Python API usage examples

## Programmatic Usage

The CLI can also be used programmatically:

```python
from mllm_project.src.pipelines import PipelineOrchestrator
from mllm_project.src.utils.config_loader import load_config_for_training

# Load configuration
config = load_config_for_training('./config')

# Create orchestrator
orchestrator = PipelineOrchestrator(
    config=config,
    output_dir='./results',
    verbose=True
)

# Run pipeline
results = orchestrator.run(
    stages=['explore', 'train', 'evaluate'],
    resume=False
)
```

## API Endpoints

When running the model serving API, the following endpoints are available:

- `GET /health`: Health check
- `POST /generate`: Single text generation
- `POST /generate_batch`: Batch generation
- `GET /stats`: API statistics
- `GET /model_info`: Model information
- `GET /docs`: Interactive API documentation

### Example API Request

```bash
curl -X POST "http://localhost:8080/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "time_series": {
      "data": [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]]
    },
    "text": {
      "prompt": "Analyze this pattern:",
      "temperature": 0.8,
      "max_length": 100
    }
  }'
```

## Output Structure

The pipeline creates the following output structure:

```
pipeline_results/
├── 01_exploration/
│   ├── exploration_results.json
│   ├── exploration_report.html
│   └── plots/
├── 02_training/
│   ├── checkpoints/
│   ├── training_summary.json
│   └── plots/
├── 03_evaluation/
│   ├── evaluation_results.json
│   ├── evaluation_report.md
│   └── plots/
├── 04_demo/
│   ├── demo_results.json
│   └── generation_samples/
└── pipeline_report.md
```

## Monitoring and Logging

### Logging Levels
Control logging verbosity:
```bash
python cli.py --verbose train ...  # Debug level
python cli.py train ...             # Info level
```

### Log Files
Specify log file:
```bash
python cli.py --log-file pipeline.log train ...
```

### MLflow Tracking
Training metrics are automatically tracked in MLflow. View the UI:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable gradient accumulation
   - Use mixed precision training

2. **Slow Training**
   - Enable distributed training
   - Use larger batch sizes
   - Check data loading bottlenecks

3. **API Connection Issues**
   - Check firewall settings
   - Verify port availability
   - Enable CORS if needed

### Debug Mode
Run with verbose output for debugging:
```bash
python cli.py --verbose pipeline --config ./config/pipeline_config.yaml
```

## Advanced Features

### Distributed Training
```bash
python cli.py train --distributed --epochs 10
```

### Custom Configurations
Override configuration parameters:
```bash
python cli.py train \
    --config-dir ./custom_config \
    --epochs 50 \
    --learning-rate 0.0001
```

### Resume Training
Continue from checkpoint:
```bash
python cli.py train --resume-from ./checkpoints/checkpoint_epoch_5.pt
```

## Support

For issues and questions:
1. Check the documentation in `docs/`
2. Review example scripts in `examples/`
3. Check logs for detailed error messages
4. Refer to the main README.md for project overview

## License

See LICENSE file in the project root.
