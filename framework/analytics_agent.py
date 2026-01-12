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
            #python_repl_tool,
            isolation_forest_tool
        ]

    def get_tools(self):    
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
        
        tools = self.get_default_tools()

        return tools