"""
Orchestrator Agent that extends LocalAgent with specialized tools for data processing and knowledge retrieval.
"""

import os

from framework.local_agent import LocalAgent
from framework.tools.worker_tools import query_knowledge_base_tool, query_gemma_cloud_tool, python_repl_tool

class OrchestratorAgent(LocalAgent):
    def get_tools(self):
        # Define specialized agents/tools here
        def list_articles_directory():
            """
            Tool to list files in the 'articles' directory.
            Contents of this directory can be found by querying the knowledge base.
            Returns:
                A string listing the files in the 'articles' directory.
            """
            articles_dir = "articles"
            try:
                files = os.listdir(articles_dir)
                return f"Files in '{articles_dir}': {', '.join(files)}"
            except FileNotFoundError:
                return f"Directory '{articles_dir}' not found."
            
        def read_article_file(file_name: str, start_character: int, end_character: int) -> str:
            """
            Tool to read the content of a specified article file.
            Article files can be text, markdown, or PDF format.
            Start character and end character difference cannot be more than 500 characters.
            To read beyond that, make multiple calls.
            
            Args:
                file_name: The name of the article file to read. This file should be located in the 'articles' directory.
                start_character: The starting character position to read from the article file.
                end_character: The ending character position to read from the article file.

            Returns:
                The content of the article file as a string.
            """
            import pymupdf4llm
            articles_dir = "articles"
            file_path = os.path.join(articles_dir, file_name)

            if end_character - start_character > 500:
                end_character = start_character + 500  # Limit to 500 characters

            try:
                content = pymupdf4llm.to_markdown(file_path)  # Ensure compatibility
                content_lines = content.splitlines()

                if start_character < 0 or end_character > len(content):
                    return f"Character range ({start_character}, {end_character}) is out of bounds for file with {len(content)} characters."

                content = content[start_character:end_character]

                return content
            except FileNotFoundError:
                return f"Article file '{file_name}' not found."
        
        def delegate_to_analytics(instructions: str) -> str:
            """
            Delegates a physics analysis task to the Analytics Agent.
            The Analytics Agent has access to training, oracle, and reporting tools.
            
            Args:
                instructions: Detailed instructions for the task. 
                              Example: "Using file *.h5, run full pipeline for mass 3.4-4.0 TeV"
            """
            # Uses the generic method from the parent class
            return self.talk_to_peer("AnalyticsAgent", instructions)

        tools = [
            delegate_to_analytics,
        ]

        tools += self.get_default_tools()
        
        return tools
    
    def get_async_tools(self):
        async_tools = [
            query_gemma_cloud_tool,
            #python_repl_tool
        ]

        if self.rag_engine_enabled:
            async_tools.append(query_knowledge_base_tool)
        
        return async_tools