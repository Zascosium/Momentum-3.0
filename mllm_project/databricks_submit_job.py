#!/usr/bin/env python3
"""
Databricks Job Submission Script
Submit training jobs to Databricks cluster via CLI
"""

import os
import json
import subprocess
import argparse
import time
from typing import Dict, Any, List, Optional

def check_databricks_cli() -> bool:
    """Check if Databricks CLI is installed and configured."""
    try:
        result = subprocess.run(['databricks', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Databricks CLI version: {result.stdout.strip()}")
        
        # Check if configured
        result = subprocess.run(['databricks', 'configure', '--list'], 
                              capture_output=True, text=True, check=True)
        print("✅ Databricks CLI is configured")
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Databricks CLI not installed or not configured")
        print("Install: pip install databricks-cli")
        print("Configure: databricks configure --token")
        return False
    except FileNotFoundError:
        print("❌ Databricks CLI not found")
        print("Install: pip install databricks-cli")
        return False

def upload_project(local_path: str, remote_path: str) -> bool:
    """Upload project to Databricks workspace."""
    print(f"📤 Uploading project from {local_path} to {remote_path}")
    
    try:
        # Remove existing remote directory
        subprocess.run(['databricks', 'workspace', 'delete', remote_path, '-r'], 
                      capture_output=True)
        
        # Upload directory
        result = subprocess.run([
            'databricks', 'workspace', 'import-dir',
            local_path, remote_path, '--overwrite'
        ], capture_output=True, text=True, check=True)
        
        print("✅ Project uploaded successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed: {e.stderr}")
        return False

def create_job_config(domains: List[str], epochs: int, batch_size: int, 
                     learning_rate: float, project_path: str) -> Dict[str, Any]:
    """Create Databricks job configuration."""
    
    # Command to run
    command = [
        "python", f"{project_path}/run_pipeline.py", "full",
        "--domains"] + domains + [
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--project-root", project_path
    ]
    
    job_config = {
        "name": f"multimodal-llm-{'-'.join(domains)}-{epochs}epochs",
        "new_cluster": {
            "spark_version": "13.3.x-gpu-ml-scala2.12",
            "node_type_id": "g4dn.xlarge",
            "num_workers": 0,
            "driver_node_type_id": "g4dn.xlarge",
            "spark_conf": {
                "spark.databricks.cluster.profile": "singleNode",
                "spark.master": "local[*]"
            },
            "custom_tags": {
                "ResourceClass": "SingleNode",
                "Project": "MultimodalLLM",
                "Domains": "-".join(domains)
            },
            "spark_env_vars": {
                "PYTHONPATH": f"{project_path}/src",
                "CUDA_VISIBLE_DEVICES": "0"
            }
        },
        "libraries": [
            {"pypi": {"package": "torch==2.0.0"}},
            {"pypi": {"package": "torchvision==0.15.0"}},
            {"pypi": {"package": "transformers==4.35.0"}},
            {"pypi": {"package": "accelerate==0.24.0"}},
            {"pypi": {"package": "datasets==2.14.0"}},
            {"pypi": {"package": "pyyaml==6.0"}},
            {"pypi": {"package": "tqdm==4.65.0"}},
            {"pypi": {"package": "matplotlib==3.7.0"}},
            {"pypi": {"package": "seaborn==0.12.0"}}
        ],
        "spark_python_task": {
            "python_file": f"{project_path}/run_pipeline.py",
            "parameters": [
                "full",
                "--domains"] + domains + [
                "--epochs", str(epochs),
                "--batch-size", str(batch_size), 
                "--learning-rate", str(learning_rate),
                "--project-root", project_path
            ]
        },
        "timeout_seconds": 14400,  # 4 hours
        "max_retries": 1,
        "email_notifications": {
            "on_start": [],
            "on_success": [],
            "on_failure": []
        }
    }
    
    return job_config

def submit_job(job_config: Dict[str, Any]) -> Optional[str]:
    """Submit job to Databricks."""
    print(f"🚀 Submitting job: {job_config['name']}")
    
    try:
        # Save config to temp file
        config_file = '/tmp/databricks_job_config.json'
        with open(config_file, 'w') as f:
            json.dump(job_config, f, indent=2)
        
        # Submit job
        result = subprocess.run([
            'databricks', 'runs', 'submit', '--json-file', config_file
        ], capture_output=True, text=True, check=True)
        
        # Parse response
        response = json.loads(result.stdout)
        run_id = response['run_id']
        
        print(f"✅ Job submitted successfully!")
        print(f"🆔 Run ID: {run_id}")
        print(f"🔗 View in UI: https://{get_databricks_host()}/jobs/{run_id}")
        
        # Clean up temp file
        os.remove(config_file)
        
        return str(run_id)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Job submission failed: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse response: {e}")
        return None

def get_databricks_host() -> str:
    """Get Databricks host from CLI config."""
    try:
        result = subprocess.run(['databricks', 'configure', '--list'], 
                              capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if 'host' in line.lower():
                return line.split()[-1].replace('https://', '')
        return "your-databricks-workspace.cloud.databricks.com"
    except:
        return "your-databricks-workspace.cloud.databricks.com"

def monitor_job(run_id: str, poll_interval: int = 30) -> bool:
    """Monitor job execution."""
    print(f"👀 Monitoring job {run_id}...")
    print("Press Ctrl+C to stop monitoring (job will continue running)")
    
    try:
        while True:
            # Get run status
            result = subprocess.run([
                'databricks', 'runs', 'get', '--run-id', run_id
            ], capture_output=True, text=True, check=True)
            
            run_info = json.loads(result.stdout)
            state = run_info['state']
            
            life_cycle_state = state['life_cycle_state']
            result_state = state.get('result_state', 'N/A')
            
            print(f"📊 Status: {life_cycle_state} | Result: {result_state}")
            
            if life_cycle_state in ['TERMINATED', 'SKIPPED', 'INTERNAL_ERROR']:
                if result_state == 'SUCCESS':
                    print("🎉 Job completed successfully!")
                    return True
                else:
                    print(f"❌ Job failed with state: {result_state}")
                    print(f"🔗 Check logs: https://{get_databricks_host()}/jobs/{run_id}")
                    return False
            
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        print("\n⚠️ Monitoring stopped (job continues running)")
        print(f"🔗 Check status: https://{get_databricks_host()}/jobs/{run_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to get job status: {e.stderr}")
        return False

def list_runs(limit: int = 10) -> None:
    """List recent job runs."""
    print(f"📋 Recent {limit} job runs:")
    
    try:
        result = subprocess.run([
            'databricks', 'runs', 'list', '--limit', str(limit)
        ], capture_output=True, text=True, check=True)
        
        runs = json.loads(result.stdout)
        
        for run in runs.get('runs', []):
            run_id = run['run_id']
            job_name = run['run_name']
            state = run['state']['life_cycle_state']
            result_state = run['state'].get('result_state', 'N/A')
            
            print(f"🆔 {run_id} | {job_name} | {state} | {result_state}")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to list runs: {e.stderr}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Databricks Job Submission for Multimodal LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training job
  python databricks_submit_job.py submit --domains Agriculture --epochs 2

  # Full training job  
  python databricks_submit_job.py submit --domains Agriculture Climate Economy --epochs 5

  # Monitor existing job
  python databricks_submit_job.py monitor --run-id 12345

  # List recent jobs
  python databricks_submit_job.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit training job')
    submit_parser.add_argument('--domains', nargs='+', default=['Agriculture'],
                              help='Domains to train on')
    submit_parser.add_argument('--epochs', type=int, default=3,
                              help='Number of training epochs')
    submit_parser.add_argument('--batch-size', type=int, default=2,
                              help='Training batch size')
    submit_parser.add_argument('--learning-rate', type=float, default=5e-5,
                              help='Learning rate')
    submit_parser.add_argument('--project-path', type=str, 
                              default='/Workspace/Repos/mllm_project',
                              help='Project path in Databricks workspace')
    submit_parser.add_argument('--local-path', type=str, default='.',
                              help='Local project path for upload')
    submit_parser.add_argument('--upload', action='store_true',
                              help='Upload project before submitting')
    submit_parser.add_argument('--monitor', action='store_true',
                              help='Monitor job after submission')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor job')
    monitor_parser.add_argument('--run-id', type=str, required=True,
                               help='Run ID to monitor')
    monitor_parser.add_argument('--poll-interval', type=int, default=30,
                               help='Polling interval in seconds')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List recent jobs')
    list_parser.add_argument('--limit', type=int, default=10,
                            help='Number of runs to list')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Check Databricks CLI
    if not check_databricks_cli():
        return
    
    if args.command == 'submit':
        # Upload project if requested
        if args.upload:
            if not upload_project(args.local_path, args.project_path):
                return
        
        # Create job config
        job_config = create_job_config(
            args.domains, args.epochs, args.batch_size, 
            args.learning_rate, args.project_path
        )
        
        # Submit job
        run_id = submit_job(job_config)
        
        if run_id and args.monitor:
            monitor_job(run_id)
    
    elif args.command == 'monitor':
        monitor_job(args.run_id, args.poll_interval)
    
    elif args.command == 'list':
        list_runs(args.limit)

if __name__ == "__main__":
    main()