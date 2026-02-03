"""Run Streamlit demo."""

#!/usr/bin/env python3
"""Launch Streamlit demo for IRL."""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch Streamlit demo."""
    demo_path = Path(__file__).parent.parent / "demo" / "app.py"
    
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        sys.exit(1)
    
    print("Launching Streamlit demo...")
    print("The demo will open in your browser at http://localhost:8501")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path), "--server.port", "8501"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching demo: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDemo stopped by user")

if __name__ == "__main__":
    main()
