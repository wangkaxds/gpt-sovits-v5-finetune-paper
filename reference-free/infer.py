"""Convenience entry point for code/infer_noref.py."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "code" / "infer_noref.py"), run_name="__main__")
