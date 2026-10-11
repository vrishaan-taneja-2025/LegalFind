"""Windows executable entry point for the locally installed LegalFind app."""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def find_project_root() -> Path:
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        candidates = (executable_dir, executable_dir.parent)
    else:
        candidates = (Path(__file__).resolve().parent,)

    for candidate in candidates:
        if (candidate / "WebAccess" / "main.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find the LegalFind project. Keep LegalFind.exe in the "
        "project's dist folder."
    )


def main() -> int:
    root = find_project_root()
    python = root / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        print("Project Python environment is missing. Run setup.bat first.")
        return 1
    url = "http://127.0.0.1:8000"
    command = [
        str(python), "-m", "uvicorn", "WebAccess.main:app",
        "--app-dir", str(root), "--host", "127.0.0.1", "--port", "8000",
    ]
    print(f"Starting LegalFind at {url}. Close this window to stop the server.")
    server = subprocess.Popen(command, cwd=root)
    try:
        time.sleep(2)
        if server.poll() is None:
            webbrowser.open(url)
        return server.wait()
    except KeyboardInterrupt:
        server.terminate()
        return server.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError) as error:
        print(error)
        raise SystemExit(1)
