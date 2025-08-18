"""
Clean Time-MMD Dataset Downloader Utility.

Downloads and sets up the Time-MMD dataset from the official repository.
Based on the official Time-MMD structure:
- numerical/{Domain}/{Domain}.csv (start_date, end_date, OT, other_variables)
- textual/{Domain}/{Domain}_report.csv (start_date, end_date, fact, pred)
- textual/{Domain}/{Domain}_search.csv (start_date, end_date, fact, pred)
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TimeMmdDownloader:
    """
    Utility to download and setup Time-MMD dataset.
    """
    
    # Official Time-MMD repository information
    GITHUB_REPO = "https://github.com/AdityaLab/Time-MMD"
    OFFICIAL_DOMAINS = [
        'Agriculture', 'Climate', 'Economy', 'Energy', 
        'Environment', 'Health', 'Security', 'Social_Good', 'Traffic'
    ]
    
    def __init__(self, data_dir: str = "./data/time_mmd"):
        """
        Initialize the downloader.
        
        Args:
            data_dir: Directory to download and store the dataset
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories following Time-MMD structure
        self.numerical_dir = self.data_dir / "numerical"
        self.textual_dir = self.data_dir / "textual"
        self.tasks_dir = self.data_dir / "Downstream_Tasks"
        
        for subdir in [self.numerical_dir, self.textual_dir, self.tasks_dir]:
            subdir.mkdir(parents=True, exist_ok=True)
    
    def create_sample_dataset(
        self, 
        domains: Optional[List[str]] = None,
        num_samples: int = 1000,
        num_reports: int = 100
    ) -> bool:
        """
        Create a sample Time-MMD dataset with the correct structure.
        
        Args:
            domains: List of domains to create (None for first 3 official domains)
            num_samples: Number of numerical samples per domain
            num_reports: Number of text reports per domain
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Creating sample Time-MMD dataset in {self.data_dir}")
        
        if domains is None:
            domains = self.OFFICIAL_DOMAINS[:3]  # Use first 3 domains for demo
        
        try:
            success = True
            success &= self._create_numerical_data(domains, num_samples)
            success &= self._create_textual_data(domains, num_reports)
            
            if success:
                self._create_dataset_info(domains)
                logger.info("Sample Time-MMD dataset created successfully!")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to create sample dataset: {e}")
            return False
    
    def _create_numerical_data(self, domains: List[str], num_samples: int) -> bool:
        """Create sample numerical data in official Time-MMD format."""
        try:
            for domain in domains:
                domain_dir = self.numerical_dir / domain
                domain_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate sample data according to Time-MMD format
                start_dates = pd.date_range(start='2020-01-01', periods=num_samples, freq='D')
                end_dates = start_dates + pd.Timedelta(days=1)
                
                # Generate domain-specific realistic patterns
                if domain == 'Agriculture':
                    # Agricultural data with seasonal patterns
                    ot_values = 100 + 10 * np.sin(np.arange(num_samples) * 2 * np.pi / 365) + np.random.normal(0, 5, num_samples)
                    other_vars = {
                        'rainfall': np.random.exponential(2, num_samples),
                        'temperature': 20 + 10 * np.sin(np.arange(num_samples) * 2 * np.pi / 365) + np.random.normal(0, 3, num_samples),
                        'soil_moisture': 30 + 10 * np.random.random(num_samples),
                        'crop_yield_index': ot_values * (0.8 + 0.4 * np.random.random(num_samples))
                    }
                elif domain == 'Climate':
                    # Climate data with weather patterns
                    ot_values = 15 + 10 * np.sin(np.arange(num_samples) * 2 * np.pi / 365) + np.random.normal(0, 2, num_samples)
                    other_vars = {
                        'humidity': 50 + 20 * np.random.random(num_samples),
                        'pressure': 1013 + np.random.normal(0, 10, num_samples),
                        'wind_speed': np.random.exponential(5, num_samples),
                        'precipitation': np.random.exponential(1, num_samples)
                    }
                elif domain == 'Economy':
                    # Economic data with market trends
                    ot_values = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, num_samples))
                    other_vars = {
                        'inflation': 2 + np.random.normal(0, 0.5, num_samples),
                        'unemployment': 5 + np.random.normal(0, 1, num_samples),
                        'gdp_growth': 2.5 + np.random.normal(0, 0.8, num_samples),
                        'market_volatility': np.random.exponential(1, num_samples)
                    }
                else:
                    # Generic pattern for other domains
                    ot_values = 50 + np.random.normal(0, 10, num_samples)
                    other_vars = {
                        'feature1': np.random.normal(10, 3, num_samples),
                        'feature2': np.random.normal(0, 1, num_samples),
                        'feature3': ot_values * 1.2 + np.random.normal(0, 2, num_samples)
                    }
                
                # Create DataFrame with official Time-MMD format
                data = {
                    'start_date': start_dates,
                    'end_date': end_dates,
                    'OT': ot_values,  # Official target column name in Time-MMD
                    **other_vars
                }
                
                df = pd.DataFrame(data)
                
                # Save to CSV with domain naming convention
                csv_path = domain_dir / f"{domain}.csv"
                df.to_csv(csv_path, index=False)
                
                logger.info(f"Created numerical data: {csv_path} ({len(df)} samples)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create numerical data: {e}")
            return False
    
    def _create_textual_data(self, domains: List[str], num_reports: int) -> bool:
        """Create sample textual data in official Time-MMD format."""
        try:
            for domain in domains:
                domain_dir = self.textual_dir / domain
                domain_dir.mkdir(parents=True, exist_ok=True)
                
                # Create report data (_report.csv)
                report_dates = pd.date_range(start='2020-01-01', periods=num_reports, freq='W')
                report_end_dates = report_dates + pd.Timedelta(days=7)
                
                # Generate domain-specific realistic text content
                facts = []
                preds = []
                
                for i in range(num_reports):
                    week_num = i + 1
                    season = ["spring", "summer", "autumn", "winter"][i % 4]
                    
                    if domain == 'Agriculture':
                        fact = f"Agricultural monitoring report week {week_num}: {season.title()} crop development shows normal growth patterns with soil conditions supporting plant health. Field observations indicate adequate nutrient levels and moisture retention across major cultivation areas."
                        pred = f"Harvest forecast for upcoming period suggests {'above-average yields' if i % 3 == 0 else 'typical seasonal performance' if i % 3 == 1 else 'below-average yields due to weather variability'} based on current growth trajectories and weather predictions."
                    elif domain == 'Climate':
                        fact = f"Climate monitoring week {week_num}: {season.title()} weather patterns recorded within expected seasonal parameters. Temperature readings, precipitation levels, and atmospheric pressure measurements align with historical averages for this time period."
                        pred = f"Weather outlook indicates {'stable conditions with minimal variation' if i % 3 == 0 else 'moderate variability with potential weather events' if i % 3 == 1 else 'significant atmospheric changes requiring monitoring'} over the forecast horizon."
                    elif domain == 'Economy':
                        fact = f"Economic analysis week {week_num}: Market performance during {season} showed sustained activity across key sectors. Consumer spending patterns, business investment levels, and employment data reflect ongoing economic dynamics consistent with seasonal trends."
                        pred = f"Economic projections suggest {'continued growth momentum with positive indicators' if i % 3 == 0 else 'stable conditions with moderate expansion' if i % 3 == 1 else 'cautious outlook with potential headwinds'} for the next reporting period."
                    else:
                        fact = f"{domain} sector analysis week {week_num}: {season.title()} operational metrics demonstrate typical performance patterns for this domain. Key performance indicators and operational data points show consistency with established trends and seasonal expectations."
                        pred = f"Operational forecast indicates {'optimized performance with efficiency gains' if i % 3 == 0 else 'steady-state operations with normal variation' if i % 3 == 1 else 'operational challenges requiring adaptive strategies'} in the upcoming cycle."
                    
                    facts.append(fact)
                    preds.append(pred)
                
                # Create report DataFrame with official format
                report_df = pd.DataFrame({
                    'start_date': report_dates,
                    'end_date': report_end_dates,
                    'fact': facts,
                    'pred': preds
                })
                
                # Save with official naming convention
                report_path = domain_dir / f"{domain}_report.csv"
                report_df.to_csv(report_path, index=False)
                
                # Create search data (_search.csv) - typically smaller dataset
                num_searches = num_reports // 2
                search_dates = pd.date_range(start='2020-01-01', periods=num_searches, freq='2W')
                search_end_dates = search_dates + pd.Timedelta(days=1)
                
                search_facts = []
                search_preds = []
                
                for i in range(num_searches):
                    search_facts.append(
                        f"{domain} research synthesis {i+1}: Comprehensive analysis of domain-specific data sources reveals key patterns and correlations. Literature review and data mining efforts have identified significant trends and relationships within the observational timeframe."
                    )
                    search_preds.append(
                        f"Research-based projection {i+1}: Evidence synthesis suggests directional trends with {'high confidence' if i % 2 == 0 else 'moderate confidence'} based on converging indicators from multiple analytical approaches and data sources."
                    )
                
                search_df = pd.DataFrame({
                    'start_date': search_dates,
                    'end_date': search_end_dates,
                    'fact': search_facts,
                    'pred': search_preds
                })
                
                # Save with official naming convention
                search_path = domain_dir / f"{domain}_search.csv"
                search_df.to_csv(search_path, index=False)
                
                logger.info(f"Created textual data: {report_path} ({len(report_df)} reports), {search_path} ({len(search_df)} searches)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create textual data: {e}")
            return False
    
    def _create_dataset_info(self, domains: List[str]) -> None:
        """Create dataset information file."""
        info = {
            'dataset': 'Time-MMD',
            'version': '1.0-sample',
            'description': 'Sample Time-MMD dataset following official structure and format',
            'reference': 'https://arxiv.org/abs/2406.08627',
            'github': self.GITHUB_REPO,
            'domains': domains,
            'structure': {
                'numerical': f"{self.numerical_dir}/<Domain>/<Domain>.csv",
                'textual_report': f"{self.textual_dir}/<Domain>/<Domain>_report.csv",
                'textual_search': f"{self.textual_dir}/<Domain>/<Domain>_search.csv",
                'tasks': str(self.tasks_dir)
            },
            'format': {
                'numerical_columns': ['start_date', 'end_date', 'OT', 'other_variables...'],
                'textual_columns': ['start_date', 'end_date', 'fact', 'pred'],
                'target_column': 'OT',
                'date_format': 'YYYY-MM-DD'
            },
            'creation_date': pd.Timestamp.now().isoformat(),
            'total_domains': len(domains),
            'official_domains': self.OFFICIAL_DOMAINS
        }
        
        info_path = self.data_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Created dataset info: {info_path}")
    
    def verify_dataset(self, domains: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Verify that the dataset follows official Time-MMD structure.
        
        Args:
            domains: List of domains to verify (None for auto-detect)
            
        Returns:
            Dictionary with comprehensive verification results
        """
        if domains is None:
            # Auto-detect available domains
            domains = []
            if self.numerical_dir.exists():
                domains = [d.name for d in self.numerical_dir.iterdir() if d.is_dir()]
        
        verification = {
            'overall_status': True,
            'structure_valid': True,
            'data_quality': True,
            'domain_details': {},
            'summary': {
                'total_domains': len(domains),
                'valid_domains': 0,
                'total_files': 0,
                'total_samples': 0,
                'issues': []
            }
        }
        
        for domain in domains:
            domain_status = {
                'numerical': {'exists': False, 'samples': 0, 'columns': [], 'has_target': False},
                'report': {'exists': False, 'samples': 0, 'columns': [], 'valid_format': False},
                'search': {'exists': False, 'samples': 0, 'columns': [], 'valid_format': False}
            }
            
            # Check numerical data
            num_file = self.numerical_dir / domain / f"{domain}.csv"
            if num_file.exists():
                try:
                    df = pd.read_csv(num_file)
                    domain_status['numerical']['exists'] = True
                    domain_status['numerical']['samples'] = len(df)
                    domain_status['numerical']['columns'] = list(df.columns)
                    domain_status['numerical']['has_target'] = 'OT' in df.columns
                    verification['summary']['total_files'] += 1
                    verification['summary']['total_samples'] += len(df)
                    
                    # Validate required columns
                    required_cols = ['start_date', 'end_date']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        verification['summary']['issues'].append(f"{domain} numerical: missing {missing_cols}")
                        verification['structure_valid'] = False
                    
                except Exception as e:
                    domain_status['numerical']['exists'] = False
                    verification['summary']['issues'].append(f"{domain} numerical: read error - {e}")
                    verification['data_quality'] = False
            else:
                verification['summary']['issues'].append(f"{domain} numerical: file not found")
                verification['structure_valid'] = False
            
            # Check report data
            report_file = self.textual_dir / domain / f"{domain}_report.csv"
            if report_file.exists():
                try:
                    df = pd.read_csv(report_file)
                    domain_status['report']['exists'] = True
                    domain_status['report']['samples'] = len(df)
                    domain_status['report']['columns'] = list(df.columns)
                    
                    required_cols = ['start_date', 'end_date', 'fact', 'pred']
                    domain_status['report']['valid_format'] = all(col in df.columns for col in required_cols)
                    
                    verification['summary']['total_files'] += 1
                    
                    if not domain_status['report']['valid_format']:
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        verification['summary']['issues'].append(f"{domain} report: missing {missing_cols}")
                        verification['structure_valid'] = False
                        
                except Exception as e:
                    verification['summary']['issues'].append(f"{domain} report: read error - {e}")
                    verification['data_quality'] = False
            
            # Check search data  
            search_file = self.textual_dir / domain / f"{domain}_search.csv"
            if search_file.exists():
                try:
                    df = pd.read_csv(search_file)
                    domain_status['search']['exists'] = True
                    domain_status['search']['samples'] = len(df)
                    domain_status['search']['columns'] = list(df.columns)
                    
                    required_cols = ['start_date', 'end_date', 'fact', 'pred']
                    domain_status['search']['valid_format'] = all(col in df.columns for col in required_cols)
                    
                    verification['summary']['total_files'] += 1
                    
                    if not domain_status['search']['valid_format']:
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        verification['summary']['issues'].append(f"{domain} search: missing {missing_cols}")
                        verification['structure_valid'] = False
                        
                except Exception as e:
                    verification['summary']['issues'].append(f"{domain} search: read error - {e}")
                    verification['data_quality'] = False
            
            verification['domain_details'][domain] = domain_status
            
            # Check if domain is valid (has numerical and at least one textual)
            has_numerical = domain_status['numerical']['exists'] and domain_status['numerical']['has_target']
            has_textual = domain_status['report']['exists'] or domain_status['search']['exists']
            
            if has_numerical and has_textual:
                verification['summary']['valid_domains'] += 1
            else:
                verification['overall_status'] = False
                if not has_numerical:
                    verification['summary']['issues'].append(f"{domain}: missing valid numerical data")
                if not has_textual:
                    verification['summary']['issues'].append(f"{domain}: missing textual data")
        
        # Overall status check
        if not verification['structure_valid'] or not verification['data_quality']:
            verification['overall_status'] = False
        
        return verification