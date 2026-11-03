from pathlib import Path
import runpy


if __name__ == "__main__":
    evaluator = Path(__file__).with_name("evaluate_model3.py")
    runpy.run_path(str(evaluator), run_name="__main__")
