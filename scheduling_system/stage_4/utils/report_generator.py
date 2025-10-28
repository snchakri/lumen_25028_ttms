"""
Feasibility Report Generator for Stage 4 Feasibility Check
Generates comprehensive HTML reports with mathematical proofs
"""

import json
import logging
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

from core.data_structures import FeasibilityOutput


class FeasibilityReportGenerator:
    """
    Generates comprehensive feasibility reports in HTML format
    Includes mathematical proofs, layer details, and cross-layer metrics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_report(self, feasibility_output: FeasibilityOutput) -> str:
        """
        Generate comprehensive HTML report for feasibility results
        
        Args:
            feasibility_output: Results from feasibility checking
            
        Returns:
            str: HTML report content
        """
        try:
            self.logger.info("Generating feasibility report")
            
            # Generate HTML report
            html_content = self._generate_html_template(feasibility_output)
            
            return html_content
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return self._generate_error_report(str(e))
    
    def _generate_html_template(self, output: FeasibilityOutput) -> str:
        """Generate HTML template for feasibility report"""
        
        # Determine status styling
        status_color = "green" if output.is_feasible else "red"
        status_text = "FEASIBLE" if output.is_feasible else "INFEASIBLE"
        status_icon = "✅" if output.is_feasible else "❌"
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stage 4 Feasibility Check Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .status {{
            font-size: 2em;
            font-weight: bold;
            margin: 20px 0;
            color: {status_color};
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            color: #666;
            margin-top: 5px;
        }}
        .layer-section {{
            margin: 30px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}
        .layer-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            font-weight: bold;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .layer-content {{
            padding: 20px;
        }}
        .layer-status {{
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .status-passed {{
            background: #d4edda;
            color: #155724;
        }}
        .status-failed {{
            background: #f8d7da;
            color: #721c24;
        }}
        .status-error {{
            background: #fff3cd;
            color: #856404;
        }}
        .mathematical-proof {{
            background: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
        }}
        .proof-theorem {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 10px;
        }}
        .proof-statement {{
            margin: 10px 0;
            font-style: italic;
        }}
        .proof-conditions {{
            margin: 10px 0;
        }}
        .proof-conclusion {{
            font-weight: bold;
            margin-top: 10px;
            color: #dc3545;
        }}
        .execution-summary {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .violations {{
            background: #fff5f5;
            border: 1px solid #fed7d7;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }}
        .violation-item {{
            margin: 5px 0;
            padding: 5px;
            background: #fed7d7;
            border-radius: 3px;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            text-align: center;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Stage 4 Feasibility Check Report</h1>
            <p>Seven-Layer Mathematical Validation System</p>
            <div class="status">
                {status_icon} {status_text}
            </div>
        </div>
        
        {self._generate_execution_summary(output)}
        
        {self._generate_cross_layer_metrics(output)}
        
        {self._generate_layer_details(output)}
        
        {self._generate_mathematical_summary(output)}
        
        <div class="timestamp">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_execution_summary(self, output: FeasibilityOutput) -> str:
        """Generate execution summary section"""
        execution_time_min = output.total_execution_time_ms / 60000
        peak_memory_gb = output.peak_memory_mb / 1024
        
        return f"""
        <div class="execution-summary">
            <h3>Execution Summary</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{execution_time_min:.2f} min</div>
                    <div class="metric-label">Total Execution Time</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{peak_memory_gb:.2f} GB</div>
                    <div class="metric-label">Peak Memory Usage</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(output.layer_results)}</div>
                    <div class="metric-label">Layers Executed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len([r for r in output.layer_results if r.is_valid()])}</div>
                    <div class="metric-label">Layers Passed</div>
                </div>
            </div>
            {f'<div class="violations"><strong>Failure Reason:</strong> {output.failure_reason}</div>' if output.failure_reason else ''}
        </div>
        """
    
    def _generate_cross_layer_metrics(self, output: FeasibilityOutput) -> str:
        """Generate cross-layer metrics section"""
        if not output.cross_layer_metrics:
            return ""
        
        metrics = output.cross_layer_metrics
        
        return f"""
        <div class="execution-summary">
            <h3>Cross-Layer Metrics</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{metrics.aggregate_load_ratio:.3f}</div>
                    <div class="metric-label">Aggregate Load Ratio (ρ)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.window_tightness_index:.3f}</div>
                    <div class="metric-label">Window Tightness Index (τ)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.conflict_density:.3f}</div>
                    <div class="metric-label">Conflict Density (δ)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.total_entities}</div>
                    <div class="metric-label">Total Entities</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.total_constraints}</div>
                    <div class="metric-label">Total Constraints</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_layer_details(self, output: FeasibilityOutput) -> str:
        """Generate detailed layer results section"""
        layer_html = '<h3>Layer Validation Results</h3>'
        
        for result in output.layer_results:
            status_class = f"status-{result.status.value}"
            
            layer_html += f"""
            <div class="layer-section">
                <div class="layer-header">
                    <span>Layer {result.layer_number}: {result.layer_name}</span>
                    <span class="layer-status {status_class}">{result.status.value.upper()}</span>
                </div>
                <div class="layer-content">
                    <p><strong>Message:</strong> {result.message}</p>
                    <p><strong>Execution Time:</strong> {result.execution_time_ms:.2f} ms</p>
                    <p><strong>Memory Used:</strong> {result.memory_used_mb:.2f} MB</p>
                    
                    {self._generate_layer_violations(result)}
                    
                    {self._generate_mathematical_proof(result.mathematical_proof) if result.mathematical_proof else ''}
                </div>
            </div>
            """
        
        return layer_html
    
    def _generate_layer_violations(self, result) -> str:
        """Generate violations section for a layer"""
        violations = []
        
        # Extract violations from details
        for key, value in result.details.items():
            if isinstance(value, dict) and "violations" in value:
                violations.extend(value["violations"])
        
        if not violations:
            return ""
        
        violation_items = ''.join([f'<div class="violation-item">{violation}</div>' for violation in violations])
        
        return f"""
        <div class="violations">
            <strong>Violations ({len(violations)}):</strong>
            {violation_items}
        </div>
        """
    
    def _generate_mathematical_proof(self, proof) -> str:
        """Generate mathematical proof section"""
        if not proof:
            return ""
        
        conditions_html = ''.join([f'<li>{condition}</li>' for condition in proof.conditions])
        
        return f"""
        <div class="mathematical-proof">
            <div class="proof-theorem">Theorem: {proof.theorem}</div>
            <div class="proof-statement">{proof.proof_statement}</div>
            <div class="proof-conditions">
                <strong>Conditions:</strong>
                <ul>{conditions_html}</ul>
            </div>
            <div class="proof-conclusion">Conclusion: {proof.conclusion}</div>
            <div><strong>Complexity:</strong> {proof.complexity}</div>
        </div>
        """
    
    def _generate_mathematical_summary(self, output: FeasibilityOutput) -> str:
        """Generate mathematical summary section"""
        if not output.mathematical_summary:
            return ""
        
        return f"""
        <div class="execution-summary">
            <h3>Mathematical Summary</h3>
            <p>{output.mathematical_summary}</p>
        </div>
        """
    
    def _generate_error_report(self, error_message: str) -> str:
        """Generate error report when report generation fails"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Stage 4 Feasibility Check Report - Error</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .error {{ color: red; background: #ffe6e6; padding: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Stage 4 Feasibility Check Report</h1>
    <div class="error">
        <h2>Report Generation Failed</h2>
        <p>Error: {error_message}</p>
    </div>
</body>
</html>
        """


