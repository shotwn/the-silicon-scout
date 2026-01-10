"""
Orchestrator Agent that extends LocalAgent with specialized tools for data processing and knowledge retrieval.
"""

import os

from framework.local_agent import LocalAgent
from framework.tools.worker_tools import query_knowledge_base_tool, query_gemma_cloud_tool, python_repl_tool

class OrchestratorAgent(LocalAgent):
    def get_tools(self):
        # Define specialized agents/tools here
        def check_if_file_exists(file_name: str) -> str:
            """
            Tool to check if a file exists in the system.
            Args:
                file_name: The name of the file to check.

            Returns:
                A string indicating whether the file exists or not.
            """
      
            exists = os.path.exists(file_name)
            return f"File '{file_name}' exists: {exists}"

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
        
        def read_any_file(file_path: str, start_character: int = 0, end_character: int = 5000) -> str:
            """
            Tool to read the content of any file. Should end with .py extension.
            Code files for available tools are Python scripts located in the 'framework/tools' directory.
            Maximum character range to read is 5000 characters. To read beyond that, make multiple calls.
            
            Args:
                file_path: The path of the code file to read.
                            Example: "framework/tools/import_and_fastjet.py"
                start_character: The starting character position to read from the utility code file.
                end_character: The ending character position to read from the utility code file.
            Returns:
                The content of the utility code file as a string.
            """
            file_path = os.path.join(file_path)
            warnings = []

            supported_extensions = ('.py', '.txt', '.md', '.markdown', '.json', '.yaml', '.yml', '.jsonl')
            if not file_path.endswith(supported_extensions):
                return f"Only {supported_extensions} code files are supported."

            if end_character - start_character > 5000:
                end_character = start_character + 5000  # Limit to 5000 characters
                warnings.append("Reading limited to 5000 characters. To read more, make multiple calls with adjusted character ranges.")

            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    content = file.read()

                if start_character < 0:
                    start_character = 0
                    warnings.append("Start character was less than 0. Adjusted to 0.")
                
                if end_character > len(content):
                    end_character = len(content)
                    warnings.append(f"End character exceeded file length. Adjusted to {len(content)}.")

                content = content[start_character:end_character]

                warning_message = "\n".join(warnings)
                if warning_message:
                    return f"```python\n{content}\n```\nWarnings:\n{warning_message}"
                else:
                    return f"```python\n{content}\n```"
            except FileNotFoundError:
                return f"Code file '{file_path}' not found."
        
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
            list_folders,
            read_any_file,
            check_if_file_exists,
        ]
        
        return tools
    
    def get_async_tools(self):
        async_tools = [
            query_gemma_cloud_tool,
            #python_repl_tool
        ]

        if self.rag_engine_enabled:
            async_tools.append(query_knowledge_base_tool)
        
        return async_tools