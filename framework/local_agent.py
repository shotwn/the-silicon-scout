"""
LocalAgent class for handling local LLM interactions with tool usage via Ollama.
This is parent class for specialized agents like OrchestratorAgent.
"""
import json
import os
import uuid
import time
import re
import ollama 
import gradio as gr 

from framework.logger import get_logger

class LocalAgent:
    def __init__(
            self, 
            model_name: str, 
            persistent_messages: list= None,
            initial_messages: list= None,
            rag_engine_enabled: bool = False
        ):
        self.initial_messages = initial_messages if initial_messages is not None else []
        self.messages = [] + self.initial_messages

        # These messages are always prepended to every new response request
        self.persistent_messages = persistent_messages if persistent_messages is not None else []

        self.model_name = model_name
        self.rag_engine_enabled = rag_engine_enabled
        self.sanitize_messages = True

        self.heavy_tools = [] 
        self.tools = self.get_tools()
        self.async_tools = self.get_async_tools()
        self.peers = {}
        self.pending_job_id = None
        self.logger = get_logger(f"Agent-{self.__class__.__name__}")

        self.num_ctx = int(os.environ.get("NUM_CTX", 32768)) # Adjust based on your VRAM
        self.context_history = self.messages.copy()
        self.context_usage = {
            "used_tokens": 0,
            "capacity": self.num_ctx,
            "percentage": 0.0,
            "last_done_reason": None
        }

    def append_message(
            self, 
            role: str, 
            content: str, 
            id: str = None,
            parent_id: str = None,
            thinking: str = "",
            tool_calls: list = None,
            tool_name: str = None,
            timestamp: float = None
        ):
        """
        Appends a message to history with auto-generated ID and timestamp.
        """
        if role not in ["user", "assistant", "tool", "system"]:
            raise ValueError(f"Invalid role '{role}'. Must be 'user', 'assistant', 'tool', or 'system'.")
        
        if role == "system":
            # Importing system messages is not supported yet
            self.logger.warning("System messages cannot be appended dynamically yet.")
            return None, None
        
        if id is None:
            id = uuid.uuid4().hex


        if timestamp is None:
            timestamp = time.time()
        
        
        msg_obj = {
            "role": role, 
            "content": content, 
            "thinking": thinking,
            "tool_calls": tool_calls if tool_calls is not None else [],
            "tool_name": tool_name,
            "id": id,
            "timestamp": timestamp
        }

        if parent_id:
            msg_obj["parent_id"] = parent_id
        
        self.messages.append(msg_obj)
        self.context_history.append(msg_obj)
        
        # Get index of the appended message
        current_msg_index = len(self.messages) - 1

        return current_msg_index, id # Return the message ID for reference
    
    def flush_messages(self):
        """
        Clears the message history.
        """
        self.messages = [] + self.initial_messages
        self.context_history = self.messages.copy()

    def save_state(self, job_id, tool_name, tool_args, initiating_message_id):
        """Persist state to disk for async processing"""
        os.makedirs("jobs/pending", exist_ok=True)
        
        state_data = {
            "job_id": job_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "messages": self.messages, 
            "initiating_message_id": initiating_message_id,
            "agent_config": {
                "base_model": self.model_name
            },
            "agent_identifier": self.__class__.__name__
        }
        
        filepath = f"jobs/pending/{job_id}.json"
        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2)

        # Wait briefly to ensure file is written
        time.sleep(1)
        return filepath

    def load_state_from_job(self, job_data):
        if (
            job_data and 
            job_data.get("original_state") and 
            job_data["original_state"].get("agent_identifier") == self.__class__.__name__
        ):
            self.logger.info(f"Restoring messages for {self.__class__.__name__} from resume data.")
            history_from_job = job_data["original_state"]["messages"]
            
            # Simple resume logic: inject tool result
            tool_call_id = job_data["original_state"].get("job_id", uuid.uuid4().hex)

            # Get the last tool call made by this agent
            last_tool_call_name = job_data["original_state"].get("tool_name", None)

            # Get the caller's id
            parent_id = job_data['original_state'].get('initating_message_id', uuid.uuid4().hex)
            
            # Note: We need to append the result to conversation
            # The last message in 'messages' should be the assistant's tool call
            # We append the result as a 'tool' role message
            
            tool_result_msg = {
                "role": "tool",
                "content": job_data["tool_result"],
                "id": tool_call_id,
                "tool_name": last_tool_call_name,
                "parent_id": parent_id
            }

            self.flush_messages()
            for msg in history_from_job:
                self.messages.append(msg)

            self.messages.append(tool_result_msg)
    
    def wait_for_tool_completion(self, job_id):
        """Poll for tool completion, then resume generation."""
        result_path = f"jobs/completed/{job_id}.json"
        self.logger.info(f"Waiting for tool job completion: {job_id}")
        while not os.path.exists(result_path):
            time.sleep(3)  # Poll every 3 seconds
        
        with open(result_path, "r") as f:
            try:
                result_data = json.load(f)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Error decoding JSON from {result_path}: {e}")
                return
        
        self.logger.info(f"Tool job {job_id} completed. Loading result.")
        
        self.load_state_from_job(result_data)

        yield from self.respond_to_tool_completion()
    
    def respond_to_tool_completion(self):
        if self.messages and self.messages[-1]['role'] == 'tool':
            # Trigger generation after loading tool result
            self.logger.info("Resuming generation after tool result...")
            resume_signal = "" # Disabled again because model thinks there is new user input
            generator = self.respond(resume_signal, message_id=uuid.uuid4().hex) # this message id is unused when content is empty
            
            # Yield this back to caller
            try:
                for parsed in generator:
                    yield parsed
            except SystemExit:
                pass

            self.pending_job_id = None
    
    def get_tools(self):
        return []
    
    def get_async_tools(self):
        return []
    
    def unload_ollama_model(self):
        """
        Unloads the current Ollama model to free up VRAM.

        Sending an empty string with keepalive=0 unloads the model.
        """
        self.logger.info(f"[System] Unloading Ollama model '{self.model_name}' to free up VRAM...")
        try:
            ollama.chat(
                model=self.model_name,
                messages=[],
                stream=False,
                keep_alive=0
            )
            self.logger.info("[System] Model unloaded successfully.")
        except Exception as e:
            self.logger.warning(f"[System] Error unloading model: {e}")
    
    def messages_to_gradio_history(self) -> list[gr.ChatMessage]:
        """
        Reconstructs the chat history from self.messages to match the 
        exact format used in the live chat_function generator.
        """        
        history = []
        for msg in self.messages:
            role = msg["role"]
            content = msg.get("content", "")
            msg_id = msg.get("id")

            if role not in ["user", "assistant", "tool"]:
                continue  # Skip system or unknown roles

            if not msg_id:
                self.logger.warning("Message without ID found in history reconstruction.")
                self.logger.warning(f"Message content: {content}")
                msg_id = uuid.uuid4().hex  # Assign a random ID if missing
            
            if role == "user":
                history.append(gr.ChatMessage(
                    role="user",
                    content=content,
                    metadata={"id": msg_id}
                ))
            elif role == "assistant":
                # Thinking Bubble
                if msg.get("thinking"):
                    history.append(gr.ChatMessage(
                        role="assistant",
                        content=msg["thinking"],
                        metadata={"title": "Thinking", "parent_id": msg_id}
                    ))
                # Tool Calls
                if msg.get("tool_calls"):
                    for tool_call in msg["tool_calls"]:
                        history.append(gr.ChatMessage(
                            role="assistant",
                            content=f"Tool Call:\nFunction: {tool_call['function']['name']}\nArguments: {tool_call['function']['arguments']}",
                            metadata={"title": "Tool Call", "parent_id": msg_id}
                        ))
                # Actual Assistant Response
                if msg.get("content"):
                    history.append(gr.ChatMessage(
                        role="assistant",
                        content=msg["content"],
                        metadata={"id": msg_id}
                    ))
            elif role == "tool":
                metadata = {"title": f"Tool: {msg.get('tool_name', 'Unknown')}", "id": msg_id}
                if msg.get("parent_id"):
                    metadata["parent_id"] = msg.get("parent_id")

                history.append(gr.ChatMessage(
                    role="assistant",
                    content=f"Tool Result:\n{content}",
                    metadata=metadata
                ))

        return history

    def respond(self, user_input:str, message_id=None) -> list:
        """
        Orchestrates the multi-step generation (Think -> Tool -> Think).
        Yields the FULL history at every step to keep the UI in sync.
        """
        max_steps = 100
        previous_steps_parsed = []
        while max_steps > 0:
            generated = self.generate_step(user_input, message_id)
            for parsed in generated:
                yield previous_steps_parsed + [parsed]
            
            # Finished the generation for this step
            previous_steps_parsed.append(parsed)

            if parsed.get("tool_calls"):
                for tool_call in parsed["tool_calls"]:
                    if not tool_call:
                        continue

                    # Parse Tool Data
                    if isinstance(tool_call, dict):
                        tool_name = tool_call["function"]["name"]
                        tool_args = tool_call["function"]["arguments"]
                    else:
                        tool_name = tool_call.function.name
                        tool_args = tool_call.function.arguments

                    # CHECK FOR ASYNC TOOL
                    async_tool_names = [tool.__name__ for tool in self.async_tools]
                    if tool_name in async_tool_names:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        job_id = f"{timestamp}_{uuid.uuid4().hex}"
                        
                        self.pending_job_id = job_id
                        self.save_state(job_id, tool_name, tool_args, initiating_message_id=parsed.get('id', uuid.uuid4().hex))
                        
                        yield previous_steps_parsed + [{
                            "content": f"Job {job_id} queued for {tool_name}. Waiting for completion...",
                            "tool_calls": None,
                            "thinking": "",
                            "tool_result": None
                        }]
                        
                        self.unload_ollama_model()

                        yield from self.wait_for_tool_completion(job_id)
                        return  # Exit after resuming from async tool

                    # Normal Sync Tool Execution
                    tool_result = self.run_tool_call(tool_call)
                    self.append_message("tool", tool_result, parent_id=message_id, tool_name=tool_name)

                    previous_steps_parsed.append({
                        "tool_result": tool_result,
                        "thinking": "",
                        "content": "",
                        "tool_call_json": None
                    })
                    user_input = None
                    max_steps -= 1
            else:
                return previous_steps_parsed

        # Handle Runaway Loops
        if max_steps <= 0:
            yield previous_steps_parsed + [{
                "content": "Error: Maximum tool call steps exceeded. Aborting to prevent infinite loop.",
                "tool_calls": None,
                "thinking": "",
                "tool_result": None
            }]
        
            self.logger.warning("Maximum tool call steps exceeded. Aborting generation to prevent infinite loop.")

    def generate_step(self, user_input:str, user_message_id=None):
        if user_input and user_input.strip() != "":
            self.append_message("user", user_input, user_message_id)

        # Prepare messages for Ollama
        # We perform the same sanitization (removing old <think> blocks)
        prompt_input_messages = []
        if self.sanitize_messages:
            # Leave last thinking block if present
            spare_thinking_count = 3 # Keep last 3 thinking blocks
            for msg in list(reversed(self.messages)):
                clean_msg = msg.copy()
                if clean_msg["role"] == "assistant" and clean_msg.get("thinking", "").strip() != "":
                    if spare_thinking_count > 0:
                        spare_thinking_count -= 1
                    else:
                        # Clean thinking blocks
                        clean_msg["thinking"] = "[Removed previous thinking to reduce context.]"
                prompt_input_messages.insert(0, clean_msg)
        else:
            prompt_input_messages = self.messages

        # Always prepend persistent (system) messages
        prompt_input_messages = self.persistent_messages + prompt_input_messages

        self.logger.info(f"> Sending request to Ollama ({self.model_name}) --->")
        
        # Log prompt
        log_root_dir = os.path.join("logs")
        current_time = time.strftime("%Y%m%d_%H%M%S")
        session_id = os.environ.get("FRAMEWORK_SESSION_ID", "default_session")
        prompt_dir = os.path.join(log_root_dir, "prompts", session_id)
        if not os.path.exists(prompt_dir):
            os.makedirs(prompt_dir, exist_ok=True)

        with open(f"{prompt_dir}/prompt_{current_time}.json", "w", encoding="utf-8") as f:
            json.dump(prompt_input_messages, f, indent=2)

        # Call Ollama API with streaming
        stream = ollama.chat(
            model=self.model_name,
            messages=prompt_input_messages,
            stream=True,
            think=True,
            tools= self.tools + self.async_tools,
            options={
                "temperature": 0.6,
                "top_p": 0.9,
                "num_ctx": self.num_ctx, # Adjust based on your VRAM
            },
            keep_alive=300 # Keep model loaded for 5 minutes this is the default
        )

        assistant_message_id = uuid.uuid4().hex
        timestamp = time.time()
        
        response = {
            'role': 'assistant',
            'content': "",
            'thinking': "",
            'tool_calls': [],
            'tool_name': None,
            'id': assistant_message_id
        }

        def serialize_response(resp):
            return {
                "role": resp["role"],
                "content": resp["content"],
                "thinking": resp["thinking"],
                "tool_calls": [{"function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in resp["tool_calls"]],
                "tool_name": resp["tool_name"],
                "id": assistant_message_id,
                "timestamp": timestamp # Timestamp will be final time
            }
        
        # Store response early so polling interfaces can see it
        current_msg_index, _ = self.append_message(**serialize_response(response))

        for chunk in stream:
            # Chunk is in format
            # model='qwen3:14b' created_at='2026-01-07T13:50:05.117069Z' 
            # done=False done_reason=None 
            # total_duration=None 
            # load_duration=None prompt_eval_count=None 
            # prompt_eval_duration=None eval_count=None eval_duration=None 
            # message=Message(role='assistant', content='', thinking='.', images=None, tool_name=None, tool_calls=None) 
            # logprobs=None

            # Update context usage on done
            if chunk.get('done'):
                # Extract token counts provided by Ollama
                prompt_tokens = chunk.get('prompt_eval_count', 0)
                eval_tokens = chunk.get('eval_count', 0)
                total_tokens = prompt_tokens + eval_tokens
                
                limit = self.num_ctx
                usage_ratio = (total_tokens / limit) * 100 if limit > 0 else 0
                done_reason = chunk.get('done_reason', 'unknown')

                # Update State
                self.context_usage = {
                    "used_tokens": total_tokens,
                    "capacity": limit,
                    "percentage": round(usage_ratio, 2),
                    "last_done_reason": done_reason
                }

                # Log for Debugging (Hidden from user, visible in logs)
                self.logger.info(
                    f"Context Stats: {total_tokens}/{limit} tokens used ({usage_ratio:.1f}%). "
                    f"Prompt: {prompt_tokens}, Gen: {eval_tokens} | Done Reason: {done_reason}"
                )
                
                # OPTIONAL: Early Warning
                if usage_ratio > 90.0:
                    self.logger.warning(f"⚠️ Context window is {usage_ratio:.1f}% full! Pruning recommended soon.")

            # Update response parts
            message = chunk.message
            response['role'] = message.role

            content_chunk = message.content or ""
            response['content'] += content_chunk
            
            thinking_chunk = message.thinking or ""
            response['thinking'] += thinking_chunk
            
            tool_calls_chunk = message.tool_calls or []
            response['tool_calls'] += tool_calls_chunk # Overwrite with latest tool calls, change it for multiple
            
            tool_name_chunk = message.tool_name
            response['tool_name'] = tool_name_chunk

            # Update the stored message
            self.messages[current_msg_index] = serialize_response(response)

            yield self.messages[current_msg_index]

        # Serialize because ToolCall objects are not JSON serializable
        serialized_response = self.messages[current_msg_index]

        # Log response
        response_logs_dir = os.path.join(log_root_dir, "responses", session_id)
        if not os.path.exists(response_logs_dir):
            os.makedirs(response_logs_dir, exist_ok=True)
            
        with open(f"{response_logs_dir}/response_{current_time}.txt", "w", encoding="utf-8") as f:
            f.write(json.dumps(serialized_response, indent=2))

        yield serialized_response

    def run_tool_call(self, tool_call_obj):
        """
        Executes a single tool call. 
        Supports both Ollama ToolCall objects and serialized dictionaries.
        """
        try:
            # Extract Name and Args based on input type
            if isinstance(tool_call_obj, dict):
                # Handle serialized dict: {'function': {'name': '...', 'arguments': ...}}
                func_data = tool_call_obj.get("function", {})
                tool_name = func_data.get("name")
                tool_args = func_data.get("arguments", {})
            elif hasattr(tool_call_obj, "function"):
                # Handle raw Ollama Object
                tool_name = tool_call_obj.function.name
                tool_args = tool_call_obj.function.arguments
            else:
                return f"Error: Unknown tool call format {type(tool_call_obj)}"

            self.logger.info(f"Executing tool call: {tool_name} with args: {tool_args}")
            
            # Find and Execute Tool
            for tool in self.tools:
                if tool.__name__ == tool_name:
                    try:
                        result = tool(**tool_args)
                        return str(result)
                    except Exception as e:
                        return f"Error executing tool {tool_name}: {e}"
            
            return f"Tool {tool_name} not found."
            
        except Exception as e:
            return f"Failed to process tool call: {str(e)}"

    def tool_result_processor(self, tool_name: str, result: str):
        result_token = "<tool_result>"
        result_end_token = "</tool_result>"
        if result_token in result and result_end_token in result:
            start_idx = result.index(result_token) + len(result_token)
            end_idx = result.index(result_end_token)
            result_content = result[start_idx:end_idx].strip()
            return result_content
        return result

    def register_peer(self, name: str, agent_instance):
        self.peers[name] = agent_instance

    def talk_to_peer(self, peer_name: str, message: str) -> str:
        if peer_name not in self.peers:
            return f"System Error: Agent '{peer_name}' is not known."

        target_agent = self.peers[peer_name]
        self.logger.info(f"\n[System] 🔄 {self.__class__.__name__} is delegating to {peer_name}...")
        task_id = uuid.uuid4().hex
        
        response_generator = target_agent.respond(message, message_id=task_id)
        final_content = "No response generated."
        
        for step in response_generator:
            last_state = step[-1]
            if last_state.get("content"):
                final_content = last_state["content"]

        self.logger.info(f"[System] ✅ {peer_name} finished task.\n")
        return f"Response from {peer_name}:\n{final_content}"
    
    def get_default_tools(self):
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
                    return f"File Contents:\n```{content}```\n\n Tool Warnings:\n{warning_message}"
                else:
                    return f"File Contents:\n```{content}```"
            except FileNotFoundError:
                return f"Code file '{file_path}' not found."
            
        def get_timestamp() -> str:
            """
            Tool to get the current timestamp.
            Returns:
                A string representing the current timestamp.
            """
            from datetime import datetime
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
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
    
        return [read_any_file, get_timestamp, list_folders, check_if_file_exists]
   