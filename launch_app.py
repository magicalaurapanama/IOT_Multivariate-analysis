"""
Launcher script for the Streamlit Anomaly Detection App

This script provides easy commands to launch the web application.
"""

import subprocess
import sys
import os

def launch_streamlit_app():
    """Launch the Streamlit application."""
    
    print("🚀 Launching IOT Anomaly Detection Web App...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        print("✅ Streamlit installed successfully")
    
    # Launch the app
    try:
        print("🌐 Opening web browser...")
        print("📱 The app will be available at: http://localhost:8501")
        print("🔄 Starting Streamlit server...")
        print("\n" + "="*50)
        print("🛑 To stop the server, press Ctrl+C")
        print("="*50 + "\n")
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print("\n💡 Try running manually:")
        print("   streamlit run streamlit_app.py")

if __name__ == "__main__":
    launch_streamlit_app()
