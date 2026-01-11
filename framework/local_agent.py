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
            initial_messages: list= None,
            rag_engine_enabled: bool = False
        ):
        self.initial_messages = initial_messages if initial_messages is not None else []
        self.messages = [] + self.initial_messages
        self.model_name = model_name
        self.rag_engine_enabled = rag_engine_enabled
        self.sanitize_messages = True

        self.heavy_tools = [] 
        self.tools = self.get_tools()
        self.async_tools = self.get_async_tools()
        self.peers = {}
        self.pending_job_id = None
        self.logger = get_logger(f"Agent-{self.__class__.__name__}")

    def save_state(self, job_id, tool_name, tool_args):
        """Persist state to disk for async processing"""
        os.makedirs("jobs/pending", exist_ok=True)
        
        state_data = {
            "job_id": job_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "messages": self.messages, 
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
            tool_call_id = uuid.uuid4().hex

            # Get the last tool call made by this agent
            last_tool_call_name = job_data["original_state"].get("tool_name", None)
            
            # Note: We need to append the result to conversation
            # The last message in 'messages' should be the assistant's tool call
            # We append the result as a 'tool' role message
            
            tool_result_msg = {
                "role": "tool",
                "content": job_data["tool_result"],
                "id": tool_call_id,
                "tool_name": last_tool_call_name,
            }

            history_from_job.append(tool_result_msg)
            
            self.messages = history_from_job
    
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
            generator = self.respond(resume_signal, message_id=uuid.uuid4().hex)
            
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
            
            if role == "user":
                history.append(gr.ChatMessage(
                    role="user",
                    content=content,
                    metadata={"id": msg_id}
                ))
            elif role == "assistant":
                if msg.get("thinking"):
                    history.append(gr.ChatMessage(
                        role="assistant",
                        content=msg["thinking"],
                        metadata={"title": "Thinking", "id": msg_id}
                    ))
                if msg.get("tool_calls"):
                    for tool_call in msg["tool_calls"]:
                        history.append(gr.ChatMessage(
                            role="assistant",
                            content=f"Tool Call:\nFunction: {tool_call['function']['name']}\nArguments: {tool_call['function']['arguments']}",
                            metadata={"title": "Tool Call", "id": msg_id}
                        ))
                if msg.get("content"):
                    history.append(gr.ChatMessage(
                        role="assistant",
                        content=msg["content"],
                        metadata={"id": msg_id}
                    ))
            elif role == "tool":
                history.append(gr.ChatMessage(
                    role="assistant",
                    content=f"Tool Result:\n{content}",
                    metadata={"title": "Tool Result", "id": msg_id}
                ))
        return history

    def respond(self, user_input:str, message_id=None) -> list:
        max_steps = 10
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

                        self.save_state(job_id, tool_name, tool_args)
                        
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
                    self.messages.append({"role": "tool", "content": tool_result, "id": message_id})

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

    def generate_step(self, user_input:str, message_id=None):
        if user_input and user_input.strip() != "":
            self.messages.append({"role": "user", "content": user_input, "id": message_id})

        # Prepare messages for Ollama
        # We perform the same sanitization (removing old <think> blocks)
        prompt_input_messages = []
        if self.sanitize_messages:
            # Leave last thinking block if present
            reversed_messages = list(reversed(self.messages))
            spare_thinking_count = 3 # Keep last 3 thinking blocks
            for msg in reversed_messages:
                clean_msg = msg.copy()
                if clean_msg["role"] == "assistant" and clean_msg.get("thinking", "").strip() != "":
                    if spare_thinking_count > 0:
                        spare_thinking_count -= 1
                    else:
                        # Clean thinking blocks
                        clean_msg["thinking"] = ""
                prompt_input_messages.append(clean_msg)
            
            prompt_input_messages = list(reversed(prompt_input_messages))
        else:
            prompt_input_messages = self.messages

        # Add tools definition to the system prompt context if needed
        # Since we use manual XML tool calling, we rely on the system prompt provided in initialization.
        # Ensure your system prompt in `__main__.py` still includes the tool definitions.

        self.logger.info(f"--- Sending request to Ollama ({self.model_name}) ---")
        
        # Log prompt
        os.makedirs("debug_logs", exist_ok=True)
        current_time = time.strftime("%Y%m%d_%H%M%S")
        with open(f"debug_logs/debug_prompt_{current_time}.json", "w") as f:
            json.dump(prompt_input_messages, f, indent=2)

        # Call Ollama API with streaming
        stream = ollama.chat(
            model=self.model_name,
            messages=prompt_input_messages,
            stream=True,
            tools= self.tools + self.async_tools,
            options={
                "temperature": 0.6,
                "top_p": 0.9,
                "num_ctx": 16384, # Adjust based on your VRAM
            },
            keep_alive=300 # Keep model loaded for 5 minutes this is the default
        )

        response = {
            'role': 'assistant',
            'content': "",
            'thinking': "",
            'tool_calls': [], # Object ToolCall instances,
            'tool_name': None,
            'id': message_id
        }

        def serialize_response(resp):
            return {
                "role": resp["role"],
                "content": resp["content"],
                "thinking": resp["thinking"],
                "tool_calls": [{"function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in resp["tool_calls"]],
                "tool_name": resp["tool_name"],
                "id": resp["id"]
            }
        
        # Store response early so polling interfaces can see it
        self.messages.append(serialize_response(response))
        current_msg_index = len(self.messages) - 1

        for chunk in stream:
            # Chunk is in format
            # model='qwen3:14b' created_at='2026-01-07T13:50:05.117069Z' 
            # done=False done_reason=None 
            # total_duration=None 
            # load_duration=None prompt_eval_count=None 
            # prompt_eval_duration=None eval_count=None eval_duration=None 
            # message=Message(role='assistant', content='', thinking='.', images=None, tool_name=None, tool_calls=None) 
            # logprobs=None
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
        with open(f"debug_logs/debug_response_{current_time}.txt", "w") as f:
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