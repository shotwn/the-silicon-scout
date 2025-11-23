from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TextIteratorStreamer
import torch
import gradio as gr
import threading
import json
import os
import sys
import subprocess
from framework.RAGEngine import RAGEngine
import uuid

max_memory = {
    0: "7.2GB",
    "cpu": "20GB"
}

class LocalAgent:
    def __init__(self, model, tokenizer, initial_messages=None, rag_engine=None):
        self.initial_messages = initial_messages if initial_messages is not None else []
        self.messages = [] + self.initial_messages
        self.model: AutoModelForCausalLM = model
        self.tokenizer = tokenizer
        self.rag_engine = rag_engine
        self.tools = self.get_tools()
    
    def get_tools(self):
        # Define specialized agents/tools here
        return []

    def respond(self, user_input:str, message_id=None, chat_history=None, depth: int = 0):
        print("---- CHAT HISTORY ----", chat_history)
        if user_input and user_input.strip() != "":
            self.messages.append({"role": "user", "content": user_input, "id": message_id})

        prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            tools=self.tools,
            add_generation_prompt=True,
        )

        # Print the final prompt for debugging
        print("Final prompt to model:")
        print(prompt)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
        )

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 32768,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        def generate():
            with torch.no_grad():
                self.model.generate(**generation_kwargs)


        thread = threading.Thread(target=generate)
        thread.start()
        
        response = ""
        for new_text in streamer:
            response += new_text
            # print("Generated so far:", response, end="\r")

            #no yield response chunks for now, just return full response at the end

        self.messages.append({"role": "assistant", "content": response, "id": message_id, "depth": depth})

        parsed = self.parse_response(response)
        print("====== Parsed response:", parsed)
        yield parsed

        if parsed["tool_call_json"] is not None:
            tool_result = self.run_tool_call(parsed)
            print("Tool result:", tool_result)
            # Append tool result to messages
            self.messages.append({"role": "tool", "content": tool_result, "id": message_id, "depth": depth})
            # Re-run the model with updated messages
            yield from self.respond(user_input="", message_id=message_id, chat_history=chat_history, depth=depth+1)
        else:
            return parsed

    def parse_response(self, response: str):
        # parsing thinking content and tool calls from the response
        thinking_token = "<think>"
        thinking_end_token = "</think>"
        thinking_content = ""

        tool_call_token = "<tool_call>"
        tool_call_end_token = "</tool_call>"
        tool_call_content = ""
        tool_call_json = None

        content = response

        if thinking_token in response and thinking_end_token in response:
            start_idx = response.index(thinking_token) + len(thinking_token)
            end_idx = response.index(thinking_end_token)
            thinking_content = response[start_idx:end_idx].strip()
            content = response[end_idx + len(thinking_end_token):].strip()

        if tool_call_token in response and tool_call_end_token in response:
            start_idx = response.index(tool_call_token) + len(tool_call_token)
            end_idx = response.index(tool_call_end_token)
            tool_call_content = response[start_idx:end_idx].strip()
            content = content[end_idx + len(tool_call_end_token):].strip()
            print("Tool call content detected:", tool_call_content)
            tool_call_json = json.loads(tool_call_content)
            print("Parsed tool call JSON:", tool_call_json)
            # Here you would handle the tool call execution

        return {
            "thinking": thinking_content, 
            "content": content, 
            "tool_call_json": tool_call_json
        }

    def run_tool_call(self, parsed_response: dict):
        # Json parse the tool call and execute the corresponding tool
        print("Executing tool call:", parsed_response["tool_call_json"])
        if not parsed_response["tool_call_json"]:
            return "No tool call to execute."
        
        try:
            tool_call_request = parsed_response["tool_call_json"]

            tool_name = tool_call_request.get("name")
            tool_args = tool_call_request.get("arguments", {})
            
            for tool in self.tools:
                print(f"Checking tool: {tool.__name__}")
                print(f"Looking for tool: {tool_name}")
                if tool.__name__ == tool_name:
                    try:
                        result = tool(**tool_args)
                        print(f"Tool {tool_name} executed with result: {result}")
                        return result
                    except Exception as e:
                        print(f"Error executing tool {tool_name}: {e}")
                        return f"Error executing tool {tool_name}: {e}"
            else:
                print(f"Tool {tool_name} not found.")
                return f"Tool {tool_name} not found."
            
        except json.JSONDecodeError:
            print("Failed to parse tool call JSON.")
            return f"Failed to parse tool call."
        


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
            import os
            exists = os.path.exists(file_name)
            return f"File '{file_name}' exists: {exists}"
        
        def run_import_and_fastjet(input_file: str, min_pt: float, size_per_row: int) -> str:
            """
            Tool to run the data loading and preprocessing using FastJet.
            Uses the 'import_and_fastjet.py' script located in 'framework/utilities'.
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
                "framework/utilities/import_and_fastjet.py",
                "--input-file", input_file,
                "--min_pt", str(min_pt),
                "--size_per_row", str(size_per_row)
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True, env=os.environ)
                return f"Data preprocessing completed successfully. Output:\n{result.stdout}"
            except subprocess.CalledProcessError as e:
                return f"Data preprocessing failed. Error:\n{e.stderr}"
            
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
            Utility code files are Python scripts located in the 'framework/utilities' directory.
            Maximum character range to read is 500 characters. To read beyond that, make multiple calls.
            
            Args:
                file_path: The path of the code file to read.
                            Example: "framework/utilities/import_and_fastjet.py"
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
        
        tools = [check_if_file_exists, run_import_and_fastjet, list_articles_directory, read_any_cwd_code, list_any_cwd_folder]
        if self.rag_engine:
            return tools + [query_knowledge_base]
        
        return tools


class Framework:
    def __init__(self, *args, **kwargs):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16  # replaces torch_dtype
        )
        

        base_model_name = kwargs.get('base_model_name', 'Qwen/Qwen3-4B')
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            max_memory=max_memory,
            quantization_config=bnb_config,  # replaces load_in_4bit argument
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Initialize RAG Engine
        self.rag_engine = RAGEngine()

        # Trigger Ingestion (Only processes new files)
        self.rag_engine.ingest_files("articles")

        self.orchestrator_agent = OrchestratorAgent(
            model=self.model,
            tokenizer=self.tokenizer,
            initial_messages=[
                {
                    "role": "system", "content": "You are an orchestrator agent part of a Particle Physics Anomaly Detection System. "
                    "Your task is to orchestrate different specialized agents to analyze data and provide insights."
                    "You use tool calls to interact with specialized agents as needed."
                    "To start, ask data file name from the user and wait for the response."
                    "After response, check if the file exists using the appropriate tool. Ask user if they want to proceed with data preprocessing using FastJet."
                }
            ],
            rag_engine=self.rag_engine
        )
    
    def chat_function(self, user_input, chat_history):
        message_id = uuid.uuid4().hex
        parsed_generator = self.orchestrator_agent.respond(user_input, message_id, chat_history)
        response = []
        for parsed in parsed_generator:
            if parsed["thinking"]:
                response.append(gr.ChatMessage(
                    role="assistant",
                    content=f"{parsed['thinking']}",
                    metadata={"title": "Thinking", "id": message_id}
                ))

            if parsed["tool_call_json"]:
                tool_name = parsed["tool_call_json"].get("name", "unknown_tool")

                response.append(gr.ChatMessage(
                    role="assistant",
                    content=f"Invoking tool... ({tool_name})",
                    metadata={"title": "Tool Call", "id": message_id}
                ))
            
            if parsed["content"]:
                response.append(gr.ChatMessage(
                    role="assistant", 
                    content=parsed["content"],
                    metadata={"id": message_id}
                ))

            yield response

    def run_interactive(self):
        with gr.Blocks() as demo:
            chatbot = gr.Chatbot()

            chatbot.clear(fn=lambda: self.orchestrator_agent.messages.clear())

            gr.ChatInterface(
                fn=self.chat_function,
                title="Interactive Chat with Orchestrator Model",
                description="Chat interface for interacting with the orchestrator model.",
                type="messages",
                chatbot=chatbot
            )

        demo.launch(share=False)

if __name__ == "__main__":
    framework = Framework(base_model_name='Qwen/Qwen3-4B')
    framework.run_interactive()
    #gen = framework.chat_function("Does the file 'events_anomalydetection_v2.h5' exist?", [])
    #print("Response chunk:", gen)