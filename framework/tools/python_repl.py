import sys
import io
import os
import builtins
import shutil
import traceback

# --- CONFIGURATION ---
ALLOWED_WRITE_DIR = os.path.abspath("./toolout/repl")  # The ONLY folder allowing writes
# ---------------------

class SafePythonREPL:
    def __init__(self):
        self.globals = {}
        self.locals = {}
        
        # Pre-import common libraries to save the LLM time
        self.run("import numpy as np")
        self.run("import pandas as pd")
        self.run("import matplotlib.pyplot as plt")
        self.run("import os")
        self.run("import torch")
        self.run("import scipy")

        # Create the allowed write directory if it doesn't exist
        os.makedirs(ALLOWED_WRITE_DIR, exist_ok=True)

    def _check_write_permission(self, path):
        """Raises error if path is not inside ALLOWED_WRITE_DIR."""
        # Convert to absolute path to prevent "../" tricks
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(ALLOWED_WRITE_DIR):
            raise PermissionError(
                f"SECURITY BLOCK: You attempted to modify a file outside '{ALLOWED_WRITE_DIR}'.\n"
                f"Target: {abs_path}\n"
                "Rule: You can READ files from anywhere, but WRITE/EXECUTE only in your own folder."
            )

    def _safe_open(self, file, mode='r', *args, **kwargs):
        """Replacement for builtins.open that checks permissions."""
        if 'w' in mode or 'a' in mode or '+' in mode or 'x' in mode:
            self._check_write_permission(file)
        return builtins.open(file, mode, *args, **kwargs)

    def _safe_remove(self, path, *args, **kwargs):
        self._check_write_permission(path)
        return os.remove(path, *args, **kwargs)

    def _safe_makedirs(self, name, mode=0o777, exist_ok=False):
        self._check_write_permission(name)
        return os.makedirs(name, mode, exist_ok)

    def run(self, code: str) -> str:
        buffer = io.StringIO()
        sys.stdout = buffer
        
        # Inject our safety wrappers into the execution scope
        # This "monkey-patches" the functions ONLY for the LLM's code
        safe_globals = self.globals.copy()
        safe_globals.update({
            "open": self._safe_open,
            "os": type('os_module', (object,), {
                # Allow reading/path utils
                "path": os.path,
                "getcwd": os.getcwd,
                "listdir": os.listdir,
                "walk": os.walk,
                "environ": os.environ,
                # Intercept dangerous write operations
                "remove": self._safe_remove,
                "makedirs": self._safe_makedirs,
                "system": lambda x: print("SECURITY: os.system() is disabled."),
                "popen": lambda x: print("SECURITY: os.popen() is disabled."),
                # Add others as needed (rename, rmdir, etc.)
            })
        })

        try:
            exec(code, safe_globals, self.locals)
            output = buffer.getvalue()
            if not output.strip():
                return "Code executed successfully. (No stdout output)"
            return output
        except Exception:
            return traceback.format_exc()
        finally:
            sys.stdout = sys.__stdout__

# Initialize once
repl_instance = SafePythonREPL()

def python_repl_tool(code: str):
    """
    Executes Python code with restricted WRITE permissions.
    - READ: Allowed everywhere (load data, configs, etc.)
    - WRITE: Allowed ONLY in './toolout/'.
    - BLOCKED: os.system, shell commands.
    """
    print(f"Worker: Executing Safe Python Code...")
    return repl_instance.run(code)