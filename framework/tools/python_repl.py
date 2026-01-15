import sys
import io
import os
import builtins
import shutil
import traceback
import argparse
import matplotlib
import matplotlib.pyplot as plt

from framework.logger import get_logger

logger = get_logger(__name__)

# --- CONFIGURATION ---
ALLOWED_WRITE_DIR = os.path.abspath("./toolout/repl")
# ---------------------

# Fix Matplotlib backend to prevent GUI errors in headless environments
matplotlib.use('Agg')

class SafeOSProxy:
    """
    Proxies the real 'os' module to allow safe attributes (like name, sep, path)
    while intercepting dangerous functions.
    """
    def __init__(self, safe_remove_func, safe_makedirs_func):
        self._safe_remove = safe_remove_func
        self._safe_makedirs = safe_makedirs_func
        self._real_os = os

    def __getattr__(self, name):
        # Block Dangerous Functions
        if name in ['system', 'popen', 'spawn', 'spawnl', 'spawnle', 'exec', 'execl']:
            raise PermissionError(f"SECURITY: os.{name}() is disabled.")
        
        # Intercept Filesystem Writes
        if name == 'remove':
            return self._safe_remove
        if name == 'makedirs':
            return self._safe_makedirs
        if name == 'mkdir':
             # Map mkdir to makedirs logic or block it if you prefer
             # Simple wrapper for now:
             return self._safe_makedirs
        
        # Allow everything else (path, sep, getcwd, environ, etc.)
        return getattr(self._real_os, name)

class SafePythonREPL:
    def __init__(self):
        # Use a SINGLE persistent scope for both globals and locals.
        # This mimics a module/script scope and fixes variable persistence.
        self.scope = {}
        
        # Initialize the security environment
        self._refresh_scope()

        # Pre-import common libraries into the persistent scope
        self.run("import numpy as np")
        self.run("import pandas as pd")
        self.run("import matplotlib.pyplot as plt")
        self.run("import scipy")

        # Create the allowed write directory
        os.makedirs(ALLOWED_WRITE_DIR, exist_ok=True)

    def _check_write_permission(self, path):
        """Raises error if path is not inside ALLOWED_WRITE_DIR."""
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(ALLOWED_WRITE_DIR):
            raise PermissionError(
                f"SECURITY BLOCK: You attempted to modify a file outside '{ALLOWED_WRITE_DIR}'.\n"
                f"Target: {abs_path}\n"
                "Rule: Write operations are restricted to the './toolout/repl/' directory."
            )

    def _safe_open(self, file, mode='r', *args, **kwargs):
        if 'w' in mode or 'a' in mode or '+' in mode or 'x' in mode:
            self._check_write_permission(file)
        return builtins.open(file, mode, *args, **kwargs)

    def _safe_remove(self, path, *args, **kwargs):
        self._check_write_permission(path)
        return os.remove(path, *args, **kwargs)

    def _safe_makedirs(self, name, mode=0o777, exist_ok=False):
        self._check_write_permission(name)
        return os.makedirs(name, mode, exist_ok)

    def _refresh_scope(self):
        """Re-injects safety wrappers in case they were overwritten."""
        self.scope.update({
            "__builtins__": __builtins__,
            "open": self._safe_open,
            # Use the robust proxy instead of a brittle dummy object
            "os": SafeOSProxy(self._safe_remove, self._safe_makedirs),
            # Helper for the LLM to know where to save
            "SAVE_DIR": ALLOWED_WRITE_DIR 
        })

    def run(self, code: str) -> str:
        buffer = io.StringIO()
        sys.stdout = buffer
        
        # Ensure security wrappers are present
        self._refresh_scope()

        try:
            # Execute in the unified persistent scope
            exec(code, self.scope, self.scope)
            
            output = buffer.getvalue()
            if not output.strip():
                return "Code executed successfully. (No stdout output)"
            return output
        except Exception:
            return traceback.format_exc()
        finally:
            sys.stdout = sys.__stdout__

# Initialize once to maintain state across calls
repl_instance = SafePythonREPL()

def python_repl_tool(code: str):
    """
    Executes Python code with persistent state and restricted write permissions.
    """
    logger.info((f"Worker: Executing Safe Python Code..."))
    return repl_instance.run(code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safe Python REPL Tool")
    parser.add_argument("--code", type=str, required=True, help="Python code to execute")
    args = parser.parse_args()
    
    result = python_repl_tool(args.code)
    logger.info((result))