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
    
        def report_to_orchestrator(self, reports: str):
            """
                Reports back to the Orchestrator Agent.
                Orchestrator is the parent agent that can give high-level instructions.
                
                Args:
                    reports: The report string to send back to the Orchestrator.
                Returns:
            """
            self.talk_to_peer(agent_name="orchestrator_agent", message=reports)

            return "Report sent to Orchestrator Agent. No further response needed."

        return [list_any_cwd_folder, report_to_orchestrator]