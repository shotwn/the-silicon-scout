"""
Analytics Agent that runs tools like LaCATHODE Trainer and Report Generator.
"""

import os
import sys
import subprocess

from framework.local_agent import LocalAgent
from framework.tools.worker_tools import fastjet_tool, lacathode_preparation_tool, \
    lacathode_training_tool, lacathode_oracle_tool, lacathode_report_generator_tool, \
    propose_signal_regions_tool, python_repl_tool, isolation_forest_tool

class AnalyticsAgent(LocalAgent):
    def get_async_tools(self):
        # Define the tools available to the Analytics Agent
        return [
            fastjet_tool,
            lacathode_preparation_tool,
            lacathode_training_tool,
            lacathode_oracle_tool,
            lacathode_report_generator_tool,
            propose_signal_regions_tool,
            python_repl_tool,
            isolation_forest_tool
        ]

    def get_tools(self):
        def list_folders(folder_path: str) -> str:
            """
            Tool to list files in any specified folder in the current working directory.
            Args:
                folder_path: The path of the folder to list files from.
            Returns:
                A string listing the files in the specified folder.
            """
            full_folder_path = os.path.join(os.getcwd(), folder_path)
            try:
                files = os.listdir(full_folder_path)
                return f"Files in '{folder_path}': {', '.join(files)}"
            except FileNotFoundError:
                return f"Directory '{folder_path}' not found."
    
        def report_to_orchestrator(reports: str):
            """
                ! There is no need for this because orchestrator can directly read the output if it called us.
                Reports back to the Orchestrator Agent.
                Orchestrator is the parent agent that can give high-level instructions.
                
                Args:
                    reports: The report string to send back to the Orchestrator.
                Returns:
            """
            self.talk_to_peer(peer_name="OrchestratorAgent", message=reports)

            return "Report sent to Orchestrator Agent. No further response needed."
        
        def get_timestamp() -> str:
            """
            Tool to get the current timestamp.
            Returns:
                A string representing the current timestamp.
            """
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return [list_folders, get_timestamp]