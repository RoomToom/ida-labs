"""Execute all notebook cells with the current Python environment."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
# Keep runtime files local instead of modifying the user's global Jupyter setup.
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".jupyter" / "runtime"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".jupyter" / "ipython"))
Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["IPYTHONDIR"]).mkdir(parents=True, exist_ok=True)
path = ROOT / "assignement-1.ipynb"
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=240, kernel_name="python3",
               resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f"Executed {sum(cell.cell_type == 'code' for cell in notebook.cells)} code cells without errors.")
