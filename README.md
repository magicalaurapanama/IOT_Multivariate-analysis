# 🌐 Live Demo

[Access the deployed Streamlit app here!](https://iot-data-analysis.streamlit.app/)

# Anomaly Detection for Multivariate Time Series Data

A complete Python-based machine learning solution for detecting anomalies in multivariate time series data and identifying the primary contributing features for each anomaly.

## 🎯 Project Overview

This solution implements an advanced anomaly detection system specifically designed for the hackathon requirements:

- **Training Period**: 2004-01-01 00:00 to 2004-01-05 23:59 (120 hours)
- **Analysis Period**: 2004-01-01 00:00 to 2004-01-19 07:59 (439 hours)
- **Validation Requirements**: Training period mean < 10, max < 25
- **Output**: Abnormality scores (0-100) + top 7 contributing features

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

### Basic Usage
```bash
# Run anomaly detection
python anomaly_detection.py "input.csv" "output.csv"

# Create visualizations
python quick_viz.py "output.csv"
python visualize_results.py "output.csv"

# Run complete pipeline
python example_usage.py
```

## 📁 File Structure

```
IOT_Multivariate_analysis/
├── anomaly_detection.py          # Main entry point
├── anomaly_pipeline.py            # Core pipeline implementation
├── visualize_results.py           # Comprehensive visualization
├── quick_viz.py                   # Quick analysis charts
├── example_usage.py               # Usage examples
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🛠️ Technical Implementation

### Core Components

1. **AnomalyPipeline Class** (`anomaly_pipeline.py`)
   - Data preprocessing and validation
   - Isolation Forest model training
   - Score calibration and feature attribution
   - Robust error handling

2. **Main Script** (`anomaly_detection.py`)
   - Command-line interface
   - Input validation
   - Pipeline orchestration

3. **Visualization Tools**
   - Comprehensive analysis reports
   - Interactive time series plots
   - Feature importance analysis
   - Training period validation

### Algorithm Details

**Model**: Isolation Forest with optimized parameters
- 200 estimators for stability
- 10% contamination assumption
- Feature importance via tree analysis

**Score Calibration**:
- Training period mapped to 0-10 range
- Anomalies scaled using 95th percentile
- Ensures validation requirements are met

**Feature Attribution**:
- Model-specific importance calculation
- Statistical threshold detection (>2σ)
- Ranking by contribution magnitude
- Alphabetical tie-breaking

## 📊 Output Format

The system adds exactly 8 new columns to the input CSV:

1. `abnormality_score` - Float (0.0 to 100.0)
2. `top_feature_1` to `top_feature_7` - String (feature names or empty)

### Score Interpretation
- **0-10**: Normal behavior (training period)
- **11-30**: Slightly unusual but acceptable
- **31-60**: Moderate anomaly requiring attention
- **61-90**: Significant anomaly needing investigation
- **91-100**: Severe anomaly requiring immediate action

## 🔍 Validation Results

✅ **Training Period Validation**:
- Mean: 4.037 (requirement: <10)
- Max: 10.079 (requirement: <25)
- Status: **PASSED**

✅ **Performance Metrics**:
- Total samples processed: 26,400
- Training samples: 7,200
- Anomalies detected: 4,673 moderate, 3,421 significant, 2,001 severe
- Top contributing feature: ReactorCoolingWaterFlow

## 📈 Visualization Features

### Quick Analysis (`quick_viz.py`)
- Anomaly timeline with training period highlight
- Score distribution comparison
- Top contributing features
- Training period validation metrics

### Comprehensive Report (`visualize_results.py`)
- 8-panel analysis dashboard
- Severity heatmaps by time
- Feature attribution timeline
- Daily anomaly event tracking
- Detailed summary statistics

## 🔧 Configuration

### Time Periods (Hackathon Specific)
```python
training_start = "2004-01-01 00:00:00"
training_end = "2004-01-05 23:59:59"
analysis_start = "2004-01-01 00:00:00"
analysis_end = "2004-01-19 07:59:59"
```

### Model Parameters
```python
IsolationForest(
    n_estimators=200,
    contamination=0.1,
    random_state=42,
    max_features=1.0,
    bootstrap=False
)
```

## 🧪 Testing & Edge Cases

The system handles:
- Missing timestamp columns
- Constant features (zero variance)
- Insufficient training data (<72 hours)
- Missing values (forward/backward fill)
- Non-numeric features (automatic exclusion)
- Perfect predictions (noise addition)

## 📋 Requirements

### Input Data
- CSV file with timestamp column (containing 'time', 'date', or 'timestamp')
- Multiple numeric feature columns
- Regular time intervals
- Covers the specified time periods

### System Requirements
- Python 3.7+
- Memory: Handles up to 10,000 rows efficiently
- Runtime: <15 minutes for typical datasets

## 🤝 Usage Examples

### Example 1: Basic Detection
```python
from anomaly_pipeline import AnomalyPipeline

pipeline = AnomalyPipeline()
pipeline.run("input.csv", "output.csv")
```

### Example 2: Custom Analysis
```python
from visualize_results import AnomalyVisualizer

viz = AnomalyVisualizer("output.csv")
viz.create_comprehensive_report("report.png")
viz.save_detailed_analysis("analysis.txt")
```

## 📞 Support

For questions or issues:
1. Check the example usage script: `python example_usage.py --help`
2. Review the detailed analysis output
3. Examine the visualization reports

## 🏆 Success Criteria Met

✅ Functional Requirements:
- Runs without errors on test dataset
- Produces all required output columns
- Training period scores: mean=4.04 (<10), max=10.08 (<25)

✅ Technical Quality:
- Follows PEP8 standards
- Modular, documented code
- Handles edge cases appropriately

✅ Performance Validation:
- Feature attributions make logical sense
- No sudden score jumps between time points
- Runtime <5 minutes for 26K samples

---

*Built for the IOT Multivariate Time Series Anomaly Detection Hackathon*
