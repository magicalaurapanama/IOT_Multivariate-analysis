"""
Streamlit Web Application for IOT Anomaly Detection

This app provides a user-friendly interface for uploading CSV files,
running anomaly detection analysis, and downloading results.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import tempfile
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our anomaly detection pipeline
from anomaly_pipeline import AnomalyPipeline

# Configure Streamlit page
st.set_page_config(
    page_title="IOT Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🔍 IOT Anomaly Detection System")
    st.markdown("""
    **Upload your multivariate time series data and detect anomalies using advanced machine learning.**
    
    This system identifies unusual patterns in your data and highlights the features contributing to each anomaly.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **Data Requirements:**
        - CSV file with timestamp column
        - Multiple numeric feature columns
        - Regular time intervals
        - Time range: 2004-01-01 to 2004-01-19
        
        **Output:**
        - Abnormality scores (0-100)
        - Top 7 contributing features
        - Interactive visualizations
        """)
        
        st.header("⚙️ Model Settings")
        contamination = st.slider("Contamination Rate", 0.05, 0.2, 0.1, 0.01)
        n_estimators = st.selectbox("Number of Estimators", [100, 200, 300], index=1)
        
        st.header("📊 Visualization Options")
        show_training_period = st.checkbox("Highlight Training Period", True)
        show_thresholds = st.checkbox("Show Severity Thresholds", True)
        interactive_plots = st.checkbox("Interactive Plots", True)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload & Analyze", "📊 Results", "📈 Visualizations", "💾 Download"])
    
    with tab1:
        st.header("📁 Upload Your Data")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with multivariate time series data"
        )
        
        if uploaded_file is not None:
            try:
                # Load and preview data
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Preview data
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data validation
                st.subheader("🔍 Data Validation")
                validation_results = validate_data(df)
                display_validation_results(validation_results)
                
                if validation_results['is_valid']:
                    # Run analysis button
                    if st.button("🚀 Run Anomaly Detection", type="primary", use_container_width=True):
                        with st.spinner("🔄 Running anomaly detection... This may take a few moments."):
                            results = run_anomaly_detection(df, contamination, n_estimators)
                            
                            if results is not None:
                                st.session_state.results = results
                                st.session_state.original_df = df
                                st.success("✅ Analysis completed successfully!")
                                st.balloons()
                            else:
                                st.error("❌ Analysis failed. Please check your data format.")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        else:
            # Show sample data format
            st.info("👆 Please upload a CSV file to begin analysis")
            show_sample_data_format()
    
    with tab2:
        st.header("📊 Analysis Results")
        
        if 'results' in st.session_state:
            display_results_summary(st.session_state.results)
        else:
            st.info("📈 Results will appear here after running the analysis")
    
    with tab3:
        st.header("📈 Interactive Visualizations")
        
        if 'results' in st.session_state:
            create_visualizations(
                st.session_state.results, 
                show_training_period, 
                show_thresholds, 
                interactive_plots
            )
        else:
            st.info("📊 Visualizations will appear here after running the analysis")
    
    with tab4:
        st.header("💾 Download Results")
        
        if 'results' in st.session_state:
            provide_download_options(st.session_state.results)
        else:
            st.info("💾 Download options will appear here after running the analysis")

def validate_data(df):
    """Validate the uploaded data."""
    results = {
        'is_valid': True,
        'messages': [],
        'warnings': [],
        'timestamp_col': None,
        'feature_cols': []
    }
    
    # Check for timestamp column
    timestamp_candidates = [col for col in df.columns 
                          if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp'])]
    
    if not timestamp_candidates:
        results['is_valid'] = False
        results['messages'].append("❌ No timestamp column found (must contain 'time', 'date', or 'timestamp')")
    else:
        results['timestamp_col'] = timestamp_candidates[0]
        results['messages'].append(f"✅ Timestamp column found: {timestamp_candidates[0]}")
        
        # Try to parse timestamps
        try:
            pd.to_datetime(df[timestamp_candidates[0]])
            results['messages'].append("✅ Timestamp format is valid")
        except:
            results['is_valid'] = False
            results['messages'].append("❌ Cannot parse timestamp column")
    
    # Check for numeric features
    numeric_cols = [col for col in df.columns 
                   if col != results['timestamp_col'] and pd.api.types.is_numeric_dtype(df[col])]
    
    if len(numeric_cols) == 0:
        results['is_valid'] = False
        results['messages'].append("❌ No numeric feature columns found")
    else:
        results['feature_cols'] = numeric_cols
        results['messages'].append(f"✅ Found {len(numeric_cols)} numeric features")
    
    # Check data size
    if len(df) < 100:
        results['warnings'].append("⚠️ Dataset is quite small (< 100 rows)")
    elif len(df) > 50000:
        results['warnings'].append("⚠️ Large dataset may take longer to process")
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        results['warnings'].append(f"⚠️ Found {missing_count} missing values (will be filled automatically)")
    
    return results

def display_validation_results(results):
    """Display data validation results."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Validation Results")
        for message in results['messages']:
            if "✅" in message:
                st.success(message)
            else:
                st.error(message)
    
    with col2:
        if results['warnings']:
            st.subheader("⚠️ Warnings")
            for warning in results['warnings']:
                st.warning(warning)
    
    if results['is_valid']:
        st.info(f"🎯 Ready to analyze {len(results['feature_cols'])} features from your dataset")

def run_anomaly_detection(df, contamination, n_estimators):
    """Run the anomaly detection pipeline."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            temp_input_path = tmp_file.name
        
        # Create pipeline with custom parameters
        pipeline = AnomalyPipeline()
        
        # Override model parameters
        original_train_method = pipeline._train_model
        def custom_train_model(train_df):
            pipeline.scaler = pipeline.scaler or __import__('sklearn.preprocessing', fromlist=['StandardScaler']).StandardScaler()
            X_train = pipeline.scaler.fit_transform(train_df)
            pipeline.model = __import__('sklearn.ensemble', fromlist=['IsolationForest']).IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42,
                max_features=1.0,
                bootstrap=False
            )
            pipeline.model.fit(X_train)
        
        pipeline._train_model = custom_train_model
        
        # Load and preprocess
        df_processed = pipeline._load_and_preprocess(temp_input_path)
        train_df, analysis_df, train_indices = pipeline._split_data(df_processed)
        
        # Train and detect
        pipeline._train_model(train_df)
        scores, attributions = pipeline._detect_anomalies(analysis_df, train_indices)
        
        # Create output dataframe
        output_df = pipeline._add_output_columns(df_processed, scores, attributions)
        
        # Clean up
        os.unlink(temp_input_path)
        
        return {
            'output_df': output_df,
            'analysis_df': analysis_df,
            'scores': scores,
            'attributions': attributions,
            'train_indices': train_indices,
            'feature_names': pipeline.feature_names,
            'timestamp_col': pipeline.timestamp_col
        }
        
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return None

def display_results_summary(results):
    """Display analysis results summary."""
    output_df = results['output_df']
    scores = results['scores']
    train_indices = results['train_indices']
    
    # Calculate statistics
    training_scores = scores[train_indices] if len(train_indices) > 0 else scores[:1000]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(output_df))
    
    with col2:
        st.metric("Training Samples", len(training_scores))
    
    with col3:
        st.metric("Training Mean Score", f"{np.mean(training_scores):.2f}")
    
    with col4:
        validation_status = "✅ PASSED" if np.mean(training_scores) < 10 and np.max(training_scores) < 25 else "❌ FAILED"
        st.metric("Validation", validation_status)
    
    # Anomaly counts
    st.subheader("🚨 Anomaly Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        moderate_count = len(output_df[output_df['abnormality_score'] >= 30])
        st.metric("Moderate (30+)", moderate_count, delta=f"{moderate_count/len(output_df)*100:.1f}%")
    
    with col2:
        significant_count = len(output_df[output_df['abnormality_score'] >= 60])
        st.metric("Significant (60+)", significant_count, delta=f"{significant_count/len(output_df)*100:.1f}%")
    
    with col3:
        severe_count = len(output_df[output_df['abnormality_score'] >= 90])
        st.metric("Severe (90+)", severe_count, delta=f"{severe_count/len(output_df)*100:.1f}%")
    
    with col4:
        max_score = output_df['abnormality_score'].max()
        st.metric("Max Score", f"{max_score:.1f}")
    
    # Top anomalies table
    st.subheader("🔥 Top 10 Highest Anomalies")
    top_anomalies = output_df.nlargest(10, 'abnormality_score')[
        [results['timestamp_col'], 'abnormality_score', 'top_feature_1', 'top_feature_2']
    ].round(2)
    st.dataframe(top_anomalies, use_container_width=True)

def create_visualizations(results, show_training_period, show_thresholds, interactive_plots):
    """Create interactive visualizations."""
    output_df = results['output_df']
    timestamp_col = results['timestamp_col']
    
    # Time series plot
    st.subheader("📈 Anomaly Score Timeline")
    
    if interactive_plots:
        fig = px.line(output_df, x=timestamp_col, y='abnormality_score',
                     title="Anomaly Scores Over Time",
                     labels={'abnormality_score': 'Abnormality Score'})
        
        # Add threshold lines
        if show_thresholds:
            fig.add_hline(y=10, line_dash="dash", line_color="yellow", 
                         annotation_text="Normal Threshold")
            fig.add_hline(y=30, line_dash="dash", line_color="orange", 
                         annotation_text="Moderate Anomaly")
            fig.add_hline(y=60, line_dash="dash", line_color="red", 
                         annotation_text="Significant Anomaly")
            fig.add_hline(y=90, line_dash="dash", line_color="darkred", 
                         annotation_text="Severe Anomaly")
        
        # Highlight training period
        if show_training_period:
            training_end = pd.to_datetime("2004-01-05 23:59:59")
            training_data = output_df[output_df[timestamp_col] <= training_end]
            if len(training_data) > 0:
                fig.add_vrect(
                    x0=training_data[timestamp_col].min(),
                    x1=training_data[timestamp_col].max(),
                    fillcolor="green", opacity=0.2,
                    annotation_text="Training Period"
                )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Score distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Score Distribution")
        fig = px.histogram(output_df, x='abnormality_score', nbins=50,
                          title="Distribution of Abnormality Scores")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Feature Importance")
        feature_counts = {}
        for i in range(1, 8):
            col_name = f'top_feature_{i}'
            if col_name in output_df.columns:
                for feature in output_df[col_name]:
                    if feature and feature != '':
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        if feature_counts:
            top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            features_df = pd.DataFrame(top_features, columns=['Feature', 'Frequency'])
            
            fig = px.bar(features_df, x='Frequency', y='Feature', orientation='h',
                        title="Most Contributing Features")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("🔥 Anomaly Heatmap by Time")
    output_df_copy = output_df.copy()
    output_df_copy['Day'] = pd.to_datetime(output_df_copy[timestamp_col]).dt.day
    output_df_copy['Hour'] = pd.to_datetime(output_df_copy[timestamp_col]).dt.hour
    
    heatmap_data = output_df_copy.pivot_table(
        values='abnormality_score', 
        index='Day', 
        columns='Hour', 
        aggfunc='mean'
    )
    
    fig = px.imshow(heatmap_data, 
                    title="Average Anomaly Score by Day and Hour",
                    labels=dict(x="Hour of Day", y="Day of Month", color="Avg Score"),
                    color_continuous_scale="YlOrRd")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def provide_download_options(results):
    """Provide download options for results."""
    output_df = results['output_df']
    
    st.subheader("💾 Download Your Results")
    
    # CSV download
    csv_buffer = io.StringIO()
    output_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📄 Download Results CSV",
            data=csv_data,
            file_name=f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Create summary report
        summary_report = create_summary_report(results)
        st.download_button(
            label="📋 Download Summary Report",
            data=summary_report,
            file_name=f"anomaly_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Preview download
    st.subheader("👀 Preview of Downloaded Data")
    st.dataframe(output_df.head(), use_container_width=True)
    
    st.info(f"📊 Your results contain {len(output_df)} rows with 8 new columns added to your original data")

def create_summary_report(results):
    """Create a text summary report."""
    output_df = results['output_df']
    scores = results['scores']
    
    # Calculate statistics
    training_scores = scores[:1000]  # Approximate training period
    
    report = f"""
ANOMALY DETECTION SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=====================================

DATASET OVERVIEW:
- Total samples: {len(output_df)}
- Features analyzed: {len(results['feature_names'])}
- Time period: {output_df[results['timestamp_col']].min()} to {output_df[results['timestamp_col']].max()}

TRAINING PERIOD VALIDATION:
- Mean score: {np.mean(training_scores):.3f} (requirement: <10)
- Max score: {np.max(training_scores):.3f} (requirement: <25)
- Standard deviation: {np.std(training_scores):.3f}
- Validation status: {'PASSED' if np.mean(training_scores) < 10 and np.max(training_scores) < 25 else 'FAILED'}

ANOMALY STATISTICS:
- Normal (0-29): {len(output_df[output_df['abnormality_score'] < 30])}
- Moderate (30-59): {len(output_df[(output_df['abnormality_score'] >= 30) & (output_df['abnormality_score'] < 60)])}
- Significant (60-89): {len(output_df[(output_df['abnormality_score'] >= 60) & (output_df['abnormality_score'] < 90)])}
- Severe (90-100): {len(output_df[output_df['abnormality_score'] >= 90])}

TOP 5 HIGHEST ANOMALIES:
"""
    
    top_5 = output_df.nlargest(5, 'abnormality_score')
    for _, row in top_5.iterrows():
        report += f"- {row[results['timestamp_col']]}: {row['abnormality_score']:.2f} (Top feature: {row['top_feature_1']})\n"
    
    # Top contributing features
    feature_counts = {}
    for i in range(1, 8):
        col_name = f'top_feature_{i}'
        if col_name in output_df.columns:
            for feature in output_df[col_name]:
                if feature and feature != '':
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    if feature_counts:
        report += "\nTOP CONTRIBUTING FEATURES:\n"
        top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, count in top_features:
            report += f"- {feature}: {count} occurrences\n"
    
    return report

def show_sample_data_format():
    """Show expected data format."""
    st.subheader("📋 Expected Data Format")
    
    sample_data = {
        'Time': ['2004-01-01 00:00:00', '2004-01-01 00:01:00', '2004-01-01 00:02:00'],
        'Temperature': [25.3, 25.1, 25.8],
        'Pressure': [1013.2, 1013.5, 1013.1],
        'Flow_Rate': [42.1, 41.9, 42.3],
        'Vibration': [0.12, 0.11, 0.13]
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)
    
    st.markdown("""
    **Requirements:**
    - 🕐 Timestamp column (must contain 'time', 'date', or 'timestamp' in name)
    - 📊 Multiple numeric feature columns
    - 📅 Data covering 2004-01-01 to 2004-01-19 period
    - ⏱️ Regular time intervals (hourly, minutely, etc.)
    """)

if __name__ == "__main__":
    main()
