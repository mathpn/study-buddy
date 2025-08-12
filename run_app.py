"""
Run script for the app.

This script launches the Streamlit application with appropriate configuration
for a local setup.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit app"""
    script_dir = Path(__file__).parent
    app_path = script_dir / "streamlit_app.py"

    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        "localhost",
        "--server.port",
        "8501",
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        "light",
        "--theme.primaryColor",
        "#FF6B6B",
        "--theme.backgroundColor",
        "#FFFFFF",
        "--theme.secondaryBackgroundColor",
        "#F0F2F6",
        "--theme.textColor",
        "#262730",
    ]

    print("Starting RAG Study Assistant...")
    print("App will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        subprocess.run(cmd, cwd=script_dir, check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error running the app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
