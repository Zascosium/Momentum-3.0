"""
Inference Pipeline

This module implements the inference pipeline that corresponds to
notebook 04_inference_demo.py functionality.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Generator
import json
import logging
import time
from datetime import datetime
import random
import numpy as np

# Databricks compatibility setup
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
    current_path = Path(__file__).parent
    while current_path != current_path.parent:
        if (current_path / 'src').exists():
            sys.path.insert(0, str(current_path / 'src'))
            break
        current_path = current_path.parent

# Core dependencies with fallbacks
try:
    import torch
except ImportError:
    torch = None

# Import project modules with fallbacks
try:
    from models.multimodal_model import MultimodalLLM
except ImportError:
    try:
        from ..models.multimodal_model import MultimodalLLM
    except ImportError:
        MultimodalLLM = None

try:
    from utils.inference_utils import create_inference_pipeline
except ImportError:
    try:
        from ..utils.inference_utils import create_inference_pipeline
    except ImportError:
        def create_inference_pipeline(*args, **kwargs):
            return None

try:
    from utils.visualization import TrainingVisualizer
except ImportError:
    try:
        from ..utils.visualization import TrainingVisualizer
    except ImportError:
        class TrainingVisualizer:
            def __init__(self, *args, **kwargs): pass
            def create_plots(self, *args, **kwargs): pass

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Pipeline for model inference and demonstration.
    """
    
    def __init__(self, config: Dict[str, Any], model_path: str, demo_dir: str):
        """
        Initialize the inference pipeline.
        
        Args:
            config: Configuration dictionary
            model_path: Path to trained model
            demo_dir: Directory for demo outputs
        """
        # Check critical dependencies
        if torch is None:
            raise ImportError("PyTorch is required for inference. Install with: pip install torch")
        if MultimodalLLM is None:
            raise ImportError("MultimodalLLM model not available. Check model imports.")
        
        self.config = config
        self.model_path = Path(model_path)
        self.demo_dir = Path(demo_dir)
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.inference_engine = None
        self.generation_history = []
        
        # Sample data for demos
        self.sample_time_series = self._generate_sample_time_series()
        self.sample_prompts = self._get_sample_prompts()
        
        # Setup visualizer
        self.plots_dir = self.demo_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        self.visualizer = TrainingVisualizer(str(self.plots_dir))
        
    def run_demo(self, num_examples: int = 10, streaming: bool = False,
                 temperature: float = 0.8) -> Dict[str, Any]:
        """
        Run standard demonstration.
        
        Args:
            num_examples: Number of examples to generate
            streaming: Whether to use streaming generation
            temperature: Generation temperature
            
        Returns:
            Dictionary containing demo results
        """
        logger.info("Starting inference demo...")
        
        # Load model
        self._load_model()
        
        results = {
            'examples': [],
            'performance_stats': {},
            'generation_params': {
                'temperature': temperature,
                'streaming': streaming,
                'num_examples': num_examples
            }
        }
        
        # Generate examples
        logger.info(f"Generating {num_examples} examples...")
        
        for i in range(num_examples):
            # Select random time series and prompt
            ts_name = random.choice(list(self.sample_time_series.keys()))
            ts_data = self.sample_time_series[ts_name]['data']
            prompt = random.choice(self.sample_prompts['analysis'])
            
            # Generate text
            start_time = time.time()
            
            if streaming:
                generated_text = self._streaming_generate(
                    time_series=ts_data,
                    prompt=prompt,
                    temperature=temperature
                )
            else:
                generated_text = self._standard_generate(
                    time_series=ts_data,
                    prompt=prompt,
                    temperature=temperature
                )
            
            generation_time = time.time() - start_time
            
            # Store result
            example = {
                'id': i,
                'time_series': ts_name,
                'prompt': prompt,
                'generated': generated_text,
                'generation_time': generation_time
            }
            
            results['examples'].append(example)
            self.generation_history.append(example)
            
            logger.info(f"Example {i+1}/{num_examples} generated in {generation_time:.3f}s")
        
        # Calculate performance statistics
        generation_times = [ex['generation_time'] for ex in results['examples']]
        results['performance_stats'] = {
            'avg_generation_time': np.mean(generation_times),
            'std_generation_time': np.std(generation_times),
            'min_generation_time': np.min(generation_times),
            'max_generation_time': np.max(generation_times),
            'total_time': sum(generation_times),
            'throughput': len(generation_times) / sum(generation_times)
        }
        
        # Save demo results
        results_path = self.demo_dir / 'demo_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Demo completed. Results saved to {results_path}")
        
        return results
    
    def run_batch_demo(self, num_examples: int = 10,
                      temperature: float = 0.8) -> List[Dict[str, Any]]:
        """
        Run batch processing demonstration.
        
        Args:
            num_examples: Number of examples to generate
            temperature: Generation temperature
            
        Returns:
            List of generation results
        """
        logger.info(f"Running batch demo with {num_examples} examples...")
        
        # Load model
        self._load_model()
        
        # Prepare batch inputs
        batch_inputs = []
        for i in range(num_examples):
            ts_name = random.choice(list(self.sample_time_series.keys()))
            ts_data = self.sample_time_series[ts_name]['data']
            prompt = random.choice(self.sample_prompts['analysis'])
            
            batch_inputs.append({
                'time_series': ts_data,
                'prompt': prompt,
                'ts_name': ts_name
            })
        
        # Batch generation
        results = []
        start_time = time.time()
        
        for input_data in batch_inputs:
            generated = self._standard_generate(
                time_series=input_data['time_series'],
                prompt=input_data['prompt'],
                temperature=temperature
            )
            
            results.append({
                'time_series': input_data['ts_name'],
                'prompt': input_data['prompt'],
                'generated': generated
            })
        
        total_time = time.time() - start_time
        
        logger.info(f"Batch processing completed in {total_time:.2f}s")
        logger.info(f"Average time per sample: {total_time/num_examples:.3f}s")
        
        return results
    
    def run_interactive(self, streaming: bool = False,
                       temperature: float = 0.8):
        """
        Run interactive demonstration mode.
        
        Args:
            streaming: Whether to use streaming generation
            temperature: Generation temperature
        """
        logger.info("Starting interactive demo...")
        
        # Load model
        self._load_model()
        
        print("\n" + "="*60)
        print("Interactive Inference Demo")
        print("="*60)
        print("\nAvailable time series:")
        for i, ts_name in enumerate(self.sample_time_series.keys(), 1):
            print(f"  {i}. {ts_name}")
        
        print("\nCommands:")
        print("  'generate' - Generate text from selected time series")
        print("  'batch' - Run batch generation")
        print("  'benchmark' - Run performance benchmark")
        print("  'quit' - Exit interactive mode")
        print("\n" + "="*60)
        
        while True:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'quit':
                    print("Exiting interactive mode...")
                    break
                
                elif command == 'generate':
                    # Select time series
                    ts_names = list(self.sample_time_series.keys())
                    print("\nSelect time series:")
                    for i, name in enumerate(ts_names, 1):
                        print(f"  {i}. {name}")
                    
                    try:
                        choice = int(input("Enter number: ")) - 1
                        if 0 <= choice < len(ts_names):
                            ts_name = ts_names[choice]
                            ts_data = self.sample_time_series[ts_name]['data']
                        else:
                            print("Invalid choice")
                            continue
                    except ValueError:
                        print("Invalid input")
                        continue
                    
                    # Get prompt
                    prompt = input("Enter prompt (or press Enter for default): ").strip()
                    if not prompt:
                        prompt = random.choice(self.sample_prompts['analysis'])
                    
                    print(f"\nGenerating with prompt: '{prompt}'...")
                    
                    # Generate
                    start_time = time.time()
                    
                    if streaming:
                        print("Generated: ", end="", flush=True)
                        generated = self._streaming_generate(
                            time_series=ts_data,
                            prompt=prompt,
                            temperature=temperature,
                            print_tokens=True
                        )
                        print()  # New line after streaming
                    else:
                        generated = self._standard_generate(
                            time_series=ts_data,
                            prompt=prompt,
                            temperature=temperature
                        )
                        print(f"Generated: {generated}")
                    
                    generation_time = time.time() - start_time
                    print(f"\nGeneration time: {generation_time:.3f}s")
                
                elif command == 'batch':
                    try:
                        num = int(input("Number of examples (default 5): ") or "5")
                    except ValueError:
                        num = 5
                    
                    print(f"\nRunning batch generation with {num} examples...")
                    results = self.run_batch_demo(num, temperature)
                    
                    print(f"\nGenerated {len(results)} examples:")
                    for i, res in enumerate(results[:3], 1):
                        print(f"\n{i}. {res['prompt'][:50]}...")
                        print(f"   → {res['generated'][:100]}...")
                
                elif command == 'benchmark':
                    print("\nRunning performance benchmark...")
                    self._run_benchmark()
                
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def _load_model(self):
        """
        Load model for inference.
        """
        if self.model is not None:
            return  # Already loaded
        
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Try to use optimized inference engine
            self.inference_engine = create_inference_pipeline(
                model_path=str(self.model_path),
                config_path=None,
                device=self.device,
                optimization_level="fast"
            )
            logger.info("Using optimized inference engine")
        except:
            # Fallback to direct model loading
            logger.info("Falling back to direct model loading")
            
            if self.model_path.is_file():
                checkpoint_path = self.model_path
            elif self.model_path.is_dir():
                checkpoint_path = self.model_path / 'best_model.pt'
                if not checkpoint_path.exists():
                    checkpoint_path = self.model_path / 'final_model.pt'
            else:
                # Create mock model for demonstration
                logger.warning("Model not found, using mock model")
                self.model = self._create_mock_model()
                return
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model_config = checkpoint.get('config', self.config)
            
            self.model = MultimodalLLM(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def _create_mock_model(self):
        """
        Create mock model for demonstration.
        """
        class MockModel:
            def generate(self, **kwargs):
                time.sleep(0.1)  # Simulate processing
                responses = [
                    "The time series shows an upward trend with periodic fluctuations.",
                    "Analysis reveals seasonal patterns with increasing amplitude.",
                    "The data exhibits volatility clustering and mean reversion.",
                    "Temporal dynamics suggest underlying cyclic behavior."
                ]
                return random.choice(responses)
        
        return MockModel()
    
    def _standard_generate(self, time_series: np.ndarray, prompt: str,
                          temperature: float = 0.8) -> str:
        """
        Standard text generation.
        """
        if self.inference_engine:
            result = self.inference_engine.generate_text(
                time_series=time_series,
                text_prompt=prompt,
                temperature=temperature,
                max_length=100
            )
            return result.generated_text if hasattr(result, 'generated_text') else str(result)
        
        elif self.model:
            if hasattr(self.model, 'generate'):
                return self.model.generate(
                    time_series=torch.tensor(time_series, dtype=torch.float32).unsqueeze(0).to(self.device),
                    text_prompt=prompt,
                    temperature=temperature
                )
            else:
                # Mock generation
                return f"{prompt} The analysis shows interesting patterns in the data."
        
        return "Model not loaded"
    
    def _streaming_generate(self, time_series: np.ndarray, prompt: str,
                           temperature: float = 0.8,
                           print_tokens: bool = False) -> str:
        """
        Streaming text generation.
        """
        # Simulate streaming generation
        response = self._standard_generate(time_series, prompt, temperature)
        words = response.split()
        
        generated = []
        for word in words:
            generated.append(word)
            if print_tokens:
                print(word, end=" ", flush=True)
                time.sleep(0.05)  # Simulate streaming delay
        
        return " ".join(generated)
    
    def _run_benchmark(self):
        """
        Run performance benchmark.
        """
        scenarios = [
            {'name': 'small', 'ts_length': 50, 'prompt_length': 10},
            {'name': 'medium', 'ts_length': 200, 'prompt_length': 25},
            {'name': 'large', 'ts_length': 500, 'prompt_length': 50}
        ]
        
        results = []
        
        for scenario in scenarios:
            print(f"\nBenchmarking {scenario['name']} scenario...")
            
            # Create test data
            ts_data = np.random.randn(scenario['ts_length'], 3)
            prompt = " ".join([f"word{i}" for i in range(scenario['prompt_length'])])
            
            # Time generation
            times = []
            for _ in range(3):  # 3 trials
                start = time.time()
                _ = self._standard_generate(ts_data, prompt)
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            results.append({
                'scenario': scenario['name'],
                'avg_time': avg_time,
                'throughput': 1.0 / avg_time
            })
            
            print(f"  Average time: {avg_time:.3f}s")
            print(f"  Throughput: {1.0/avg_time:.2f} samples/sec")
        
        return results
    
    def _generate_sample_time_series(self) -> Dict[str, Dict]:
        """
        Generate sample time series for demonstration.
        """
        samples = {}
        t = np.linspace(0, 10, 200)
        
        # Trend pattern
        samples['trend'] = {
            'data': (2 * t + 0.5 * np.sin(2 * np.pi * t) + np.random.normal(0, 0.2, len(t))).reshape(-1, 1),
            'description': 'Upward trend with oscillations'
        }
        
        # Seasonal pattern
        samples['seasonal'] = {
            'data': (5 + 3 * np.sin(2 * np.pi * t / 2) + np.random.normal(0, 0.3, len(t))).reshape(-1, 1),
            'description': 'Strong seasonal pattern'
        }
        
        # Volatile pattern
        samples['volatile'] = {
            'data': np.cumsum(np.random.normal(0.001, 0.02, len(t))).reshape(-1, 1) + 100,
            'description': 'High volatility financial series'
        }
        
        # Multivariate
        samples['multivariate'] = {
            'data': np.column_stack([
                2 * t + np.random.normal(0, 0.5, len(t)),
                3 * np.sin(2 * np.pi * t / 3) + np.random.normal(0, 0.3, len(t)),
                np.cumsum(np.random.normal(0, 0.1, len(t)))
            ]),
            'description': 'Multi-dimensional correlated series'
        }
        
        return samples
    
    def _get_sample_prompts(self) -> Dict[str, List[str]]:
        """
        Get sample prompts for different scenarios.
        """
        return {
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
                "Projected trends:"
            ],
            'explanation': [
                "This time series represents:",
                "The underlying process:",
                "Key characteristics include:",
                "Pattern interpretation:"
            ]
        }
