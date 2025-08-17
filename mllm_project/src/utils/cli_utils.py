"""
CLI Utilities

This module provides utility functions for the command-line interface.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import click
from datetime import datetime
import os
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
import json
import yaml


# Rich console for pretty printing
console = Console()


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """
    Setup logging configuration with Rich handler.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    # Create handlers
    handlers = []
    
    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True
    )
    rich_handler.setLevel(level)
    handlers.append(rich_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )
    
    # Set levels for specific loggers
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)


def print_banner():
    """Print CLI banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                   Multimodal LLM Pipeline                    ║
║                                                              ║
║  Unified CLI for training and deploying multimodal models   ║
║  combining time series and text data                        ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold cyan"))


def validate_paths(paths: Dict[str, str]) -> bool:
    """
    Validate that required paths exist.
    
    Args:
        paths: Dictionary of path names to path strings
        
    Returns:
        True if all paths are valid
    """
    all_valid = True
    
    for name, path_str in paths.items():
        if path_str:
            path = Path(path_str)
            if not path.exists():
                console.print(f"[red]✗[/red] {name} not found: {path}")
                all_valid = False
            else:
                console.print(f"[green]✓[/green] {name} found: {path}")
    
    return all_valid


def create_progress_bar(description: str = "Processing") -> Progress:
    """
    Create a Rich progress bar.
    
    Args:
        description: Progress bar description
        
    Returns:
        Progress object
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    )


def print_results_table(results: Dict[str, Any], title: str = "Results"):
    """
    Print results in a formatted table.
    
    Args:
        results: Dictionary of results
        title: Table title
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    for key, value in results.items():
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        elif isinstance(value, dict):
            value_str = json.dumps(value, indent=2)
        else:
            value_str = str(value)
        
        table.add_row(key, value_str)
    
    console.print(table)


def print_stage_header(stage: str):
    """
    Print a formatted stage header.
    
    Args:
        stage: Stage name
    """
    header = f"[bold cyan]{'='*60}[/bold cyan]\n"
    header += f"[bold white]Stage: {stage.upper()}[/bold white]\n"
    header += f"[bold cyan]{'='*60}[/bold cyan]"
    console.print(header)


def print_stage_summary(stage: str, status: str, metrics: Optional[Dict[str, Any]] = None):
    """
    Print stage completion summary.
    
    Args:
        stage: Stage name
        status: Completion status
        metrics: Optional metrics to display
    """
    if status == "success":
        status_icon = "[green]✓[/green]"
        status_text = "[green]SUCCESS[/green]"
    elif status == "failed":
        status_icon = "[red]✗[/red]"
        status_text = "[red]FAILED[/red]"
    else:
        status_icon = "[yellow]⚠[/yellow]"
        status_text = "[yellow]WARNING[/yellow]"
    
    console.print(f"\n{status_icon} Stage '{stage}' completed: {status_text}")
    
    if metrics:
        for key, value in metrics.items():
            console.print(f"  • {key}: {value}")


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Ask for user confirmation.
    
    Args:
        message: Confirmation message
        default: Default response
        
    Returns:
        User's confirmation
    """
    return click.confirm(message, default=default)


def select_from_list(options: List[str], prompt: str = "Select an option") -> str:
    """
    Let user select from a list of options.
    
    Args:
        options: List of options
        prompt: Selection prompt
        
    Returns:
        Selected option
    """
    console.print(f"\n[bold]{prompt}:[/bold]")
    for i, option in enumerate(options, 1):
        console.print(f"  {i}. {option}")
    
    while True:
        try:
            choice = int(input("\nEnter number: ")) - 1
            if 0 <= choice < len(options):
                return options[choice]
            else:
                console.print("[red]Invalid choice. Please try again.[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    console.print(f"[green]✓[/green] Configuration loaded from {config_path}")
    
    return config


def save_json_results(results: Dict[str, Any], output_path: str):
    """
    Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    console.print(f"[green]✓[/green] Results saved to {output_path}")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_error(message: str, exception: Optional[Exception] = None):
    """
    Print error message.
    
    Args:
        message: Error message
        exception: Optional exception object
    """
    console.print(f"[red bold]ERROR:[/red bold] {message}")
    if exception:
        console.print(f"[red]Details: {str(exception)}[/red]")


def print_warning(message: str):
    """
    Print warning message.
    
    Args:
        message: Warning message
    """
    console.print(f"[yellow bold]WARNING:[/yellow bold] {message}")


def print_info(message: str):
    """
    Print info message.
    
    Args:
        message: Info message
    """
    console.print(f"[blue]ℹ[/blue] {message}")


def print_success(message: str):
    """
    Print success message.
    
    Args:
        message: Success message
    """
    console.print(f"[green]✓[/green] {message}")


def create_directory_structure(base_dir: str, structure: Dict[str, Any]):
    """
    Create directory structure.
    
    Args:
        base_dir: Base directory path
        structure: Directory structure as nested dict
    """
    base_path = Path(base_dir)
    
    def create_dirs(parent: Path, struct: Dict[str, Any]):
        for name, content in struct.items():
            path = parent / name
            if isinstance(content, dict):
                path.mkdir(parents=True, exist_ok=True)
                create_dirs(path, content)
            else:
                path.mkdir(parents=True, exist_ok=True)
    
    create_dirs(base_path, structure)
    console.print(f"[green]✓[/green] Directory structure created at {base_path}")


def monitor_gpu_usage():
    """
    Monitor and display GPU usage.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            console.print(f"\n[bold]GPU Information:[/bold]")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                
                console.print(f"  GPU {i}: {device_name}")
                console.print(f"    • Allocated: {memory_allocated:.2f} GB")
                console.print(f"    • Reserved: {memory_reserved:.2f} GB")
        else:
            console.print("[yellow]No GPU available[/yellow]")
    except ImportError:
        console.print("[yellow]PyTorch not available for GPU monitoring[/yellow]")


def print_config_summary(config: Dict[str, Any]):
    """
    Print configuration summary.
    
    Args:
        config: Configuration dictionary
    """
    table = Table(title="Configuration Summary", show_header=True, header_style="bold magenta")
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("Key", style="green")
    table.add_column("Value", style="white")
    
    def add_config_items(section: str, items: Dict[str, Any], prefix: str = ""):
        for key, value in items.items():
            if isinstance(value, dict):
                add_config_items(section, value, f"{prefix}{key}.")
            else:
                table.add_row(section, f"{prefix}{key}", str(value))
    
    for section, content in config.items():
        if isinstance(content, dict):
            add_config_items(section, content)
        else:
            table.add_row(section, "", str(content))
    
    console.print(table)


class CLIProgressTracker:
    """
    Progress tracker for long-running operations.
    """
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            description: Progress description
        """
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        )
        self.task = None
    
    def __enter__(self):
        """Enter context manager."""
        self.progress.__enter__()
        self.task = self.progress.add_task(self.description, total=self.total_steps)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.progress.__exit__(exc_type, exc_val, exc_tb)
    
    def update(self, step: int = 1, description: Optional[str] = None):
        """
        Update progress.
        
        Args:
            step: Number of steps to advance
            description: Optional new description
        """
        self.current_step += step
        if description:
            self.progress.update(self.task, advance=step, description=description)
        else:
            self.progress.advance(self.task, step)
    
    def set_description(self, description: str):
        """
        Set progress description.
        
        Args:
            description: New description
        """
        self.progress.update(self.task, description=description)
