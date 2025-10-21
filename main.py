import sys
import yaml
import nbformat
from pathlib import Path
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(notebook_path, timeout=600):
    """
    Executes a Jupyter notebook and overwites it with outputs.
    """
    notebook_path = Path(notebook_path)
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
    ep.preprocess(nb, {"metadata":{"path": notebook_path.parent}})

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Executed Notebook: {notebook_path}")

if __name__ == "__main__":

    import yaml

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    notebooks = config["main"]["notebooks"]
    timeout = config["main"]["timeout"]
    
    if timeout is None or str(timeout).lower() == "inf":
        timeout = None
    
    for nb in notebooks:
        run_notebook(nb)