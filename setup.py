import subprocess
import sys
import venv
from pathlib import Path

root = Path(__file__).resolve().parent
venv_dir = root / ".venv"
py = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

if not py.exists():
    venv.create(venv_dir, with_pip=True)

for cmd in (["-m", "pip", "install", "-r", "requirements.txt"],
            ["tools/download_models.py"],
            ["tools/check_env.py"]):
    subprocess.run([str(py), *cmd], cwd=root, check=True)

activate = r".venv\Scripts\activate" if sys.platform == "win32" else "source .venv/bin/activate"
print(f"\nPronto. Ative o ambiente com: {activate}")
