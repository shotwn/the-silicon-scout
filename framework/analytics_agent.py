"""
Analytics Agent that runs tools like LaCATHODE Trainer and Report Generator.
"""

import os
import sys
import subprocess

from framework.local_agent import LocalAgent
from framework.tools.worker_tools import fastjet_tool, lacathode_preparation_tool, \
    lacathode_training_tool, lacathode_oracle_tool, lacathode_report_generator_tool

class AnalyticsAgent(LocalAgent):
    def get_async_tools(self):
        # Define the tools available to the Analytics Agent
        return [
            fastjet_tool,
            lacathode_preparation_tool,
            lacathode_training_tool,
            lacathode_oracle_tool,
            lacathode_report_generator_tool,
        ]

    def get_tools(self):
        def list_any_cwd_folder(folder_path: str) -> str:
            """
            Tool to list files in any specified folder in the current working directory.
            Args:
                folder_path: The path of the folder to list files from.
            Returns:
                A string listing the files in the specified folder.
            """
            folder_path = os.path.join(os.getcwd(), folder_path)
            try:
                files = os.listdir(folder_path)
                return f"Files in '{folder_path}': {', '.join(files)}"
            except FileNotFoundError:
                return f"Directory '{folder_path}' not found."
    
        return [list_any_cwd_folder]