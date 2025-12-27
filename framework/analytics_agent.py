"""
Analytics Agent that runs tools like LaCATHODE Trainer and Report Generator.
"""

import os
import sys
import subprocess

from framework.local_agent import LocalAgent
from framework.tools.worker_tools import fastjet_tool

class AnalyticsAgent(LocalAgent):
    def get_async_tools(self):
        # Define the tools available to the Analytics Agent
        return [fastjet_tool]
