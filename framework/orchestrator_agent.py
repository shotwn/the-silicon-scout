"""
Orchestrator Agent that extends LocalAgent with specialized tools for data processing and knowledge retrieval.
"""

import os
import sys
import subprocess

from framework.local_agent import LocalAgent

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
        
        def run_import_and_fastjet(input_file: str, min_pt: float, size_per_row: int) -> str:
            """
            Tool to run the data loading and preprocessing using FastJet.
            Uses the 'import_and_fastjet.py' script located in 'framework/tools'.
            Code can be read using the 'read_utility_code' tool with the file name 'import_and_fastjet.py'.

            Args:
                input_file: The name of the data file to process. Should be in .h5 format.
                min_pt: The minimum transverse momentum for jets. Default is 20.0.
                size_per_row: Number of data columns per row (excluding label). Default is 2100 as per LHCO2020 dataset.
            Returns:
                A string summarizing the preprocessing results.
            """
            command = [
                f"{sys.executable}",
                "framework/tools/import_and_fastjet.py",
                "--input_file", input_file,
                "--min_pt", str(min_pt),
                "--size_per_row", str(size_per_row)
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True, env=os.environ)
                result_content = self.tool_result_processor("import_and_fastjet", result.stdout)
                return f"Data preprocessing completed successfully. Output:\n{result_content}"
            except subprocess.CalledProcessError as e:
                return f"Data preprocessing failed. Error:\n{e.stderr}"
            
        def lacathode_preperation(
                run_mode: str = "training",
                input_background: str = None,
                input_signal: str = None,
                input_unlabelled: str = None, 
                output_dir: str = None,
                training_fraction: float = 0.33,
                validation_fraction: float = 0.33
        ) -> str:
            """
            Tool to run the LACathode data preparation script.
            Input files are results of run_import_fastjet tool. JSONL format.
            Training mode requires both background and signal files.
            Inference mode requires unlabelled data file.
            Code available at framework/tools/lacathode_preperation.py
            Args:
                run_mode: "training" or "inference"
                input_background: Path to background data file (required for training mode)
                input_signal: Path to signal data file (required for training mode)
                input_unlabelled: Path to unlabelled data file (required for inference mode)
                output_dir: Directory to save prepared data (default: "lacathode_input_data")
                training_fraction: Fraction of data to use for training (default 0.33)
                validation_fraction: Fraction of data to use for validation (default 0.33)
            Returns:
                A string summarizing the data preparation results.
            """
            command = [
                f"{sys.executable}",
                "framework/tools/lacathode_preperation.py",
                "--run_mode", run_mode,
            ]
            if input_background:
                command += ["--input_background", input_background]
            if input_signal:
                command += ["--input_signal", input_signal]
            if input_unlabelled:
                command += ["--input_unlabelled", input_unlabelled]
            if output_dir:
                command += ["--output_dir", output_dir]
            command += [
                "--training_fraction", str(training_fraction),
                "--validation_fraction", str(validation_fraction)
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True, env=os.environ)
                result_content = self.tool_result_processor("lacathode_preperation", result.stdout)
                return f"LACathode data preparation completed successfully. Output:\n{result_content}"
            except subprocess.CalledProcessError as e:
                return f"LACathode data preparation failed. Error:\n{e.stderr}"


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
        
        def read_any_cwd_code(file_path: str, start_character: int = 0, end_character: int = 500) -> str:
            """
            Tool to read the content of any cwd code file. Should end with .py extension.
            Code files for available tools are Python scripts located in the 'framework/tools' directory.
            Maximum character range to read is 500 characters. To read beyond that, make multiple calls.
            
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

            if not file_path.endswith(".py"):
                return "Only Python (.py) code files are supported."

            if end_character - start_character > 500:
                end_character = start_character + 500  # Limit to 500 characters
                warnings.append("Reading limited to 500 characters. To read more, make multiple calls with adjusted character ranges.")

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
            
        def query_knowledge_base(search_query: str) -> str:
            """
            Searches the internal knowledge base (PDFs/Articles) for specific information.
            Args:
                search_query: The topic or question to search for. 
                              Example: "What is the mass of the Top Quark?"
            """
            print(f"RAG Query: {search_query}")
            return self.rag_engine.query(search_query)
        
        tools = [
            check_if_file_exists, 
            run_import_and_fastjet, 
            list_articles_directory, 
            read_any_cwd_code, 
            list_any_cwd_folder, 
            lacathode_preperation
        ]
        if self.rag_engine:
            tools = tools + [query_knowledge_base]
        
        # Tools that need Model to unload during their execution
        self.heavy_tools = ['run_import_and_fastjet', 'lacathode_preperation']
        
        return tools
