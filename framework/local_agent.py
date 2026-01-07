"""
LocalAgent class for handling local LLM interactions with tool usage.
This is parent class for specialized agents like OrchestratorAgent.
"""
import threading
import json
import gc
import torch
import os
import uuid
import time
import re

from transformers import AutoModelForCausalLM, TextIteratorStreamer

from framework.rag_engine import RAGEngine

class LocalAgent:
    def __init__(
            self, 
            model, 
            tokenizer, 
            initial_messages: list= None,
            rag_engine: RAGEngine = None, 
            model_loader: callable = None,
            model_unloader: callable = None
        ):
        self.initial_messages = initial_messages if initial_messages is not None else []
        self.messages = [] + self.initial_messages
        self.model: AutoModelForCausalLM = model
        self.tokenizer = tokenizer
        self.rag_engine = rag_engine
        self.model_loader = model_loader
        self.model_unloader = model_unloader
        self.sanitize_messages = True

        # Make sure only one thread accesses the model at a time
        self.model_lock = threading.Lock()

        self.heavy_tools = []  # To be populated with get_tools or manually
        self.tools = self.get_tools()
        self.async_tools = self.get_async_tools()
        self.peers = {}

    def save_state(self, job_id, tool_name, tool_args):
        """Persist state to disk for async processing"""
        os.makedirs("jobs/pending", exist_ok=True)
        
        state_data = {
            "job_id": job_id,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "messages": self.messages, # Saves full conversation context
            "agent_config": {
                "base_model": "Qwen/Qwen3-4B" # Add relevant config to restore later
            },
            "agent_identifier": self.__class__.__name__
        }
        
        filepath = f"jobs/pending/{job_id}.json"
        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2)
        return filepath
    
    def get_tools(self):
        # Define specialized agents/tools here
        return []
    
    def get_async_tools(self):
        return []
    
    def messages_to_gradio_history(self):
        """
        Reconstructs the chat history from self.messages to match the 
        exact format used in the live chat_function generator.
        """
        import gradio as gr # Imported locally to avoid dependency issues if not at top level
        
        history = []
        
        for msg in self.messages:
            role = msg["role"]
            content = msg.get("content", "")
            msg_id = msg.get("id")
            
            if role == "user":
                history.append(gr.ChatMessage(
                    role="user",
                    content=content,
                    metadata={"id": msg_id} # Keep message ID for reference
                ))
                
            elif role == "assistant":
                # We need to parse the raw stored text to separate Thinking vs Content vs Tool Calls
                # so the history looks identical to the live stream.
                parsed = self.parse_response(content)
                
                # 1. Recreate Thinking Bubble
                if parsed.get("thinking"):
                    history.append(gr.ChatMessage(
                        role="assistant",
                        content=parsed["thinking"],
                        metadata={"title": "Thinking", "id": msg_id}
                    ))
                
                # 2. Recreate Tool Call Bubble
                if parsed.get("tool_call_json"):
                    tool_name = parsed["tool_call_json"].get("name", "unknown_tool")
                    history.append(gr.ChatMessage(
                        role="assistant",
                        content=f"Invoking tool... ({tool_name})",
                        metadata={"title": "Tool Call", "id": msg_id}
                    ))
                
                # 3. Recreate Standard Content
                if parsed.get("content"):
                    history.append(gr.ChatMessage(
                        role="assistant",
                        content=parsed["content"],
                        metadata={"id": msg_id}
                    ))
                    
            elif role == "tool":
                # Recreate Tool Result Bubble
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

            # Continue if there is a tool call
            if parsed["tool_call_json"] is not None:
                tool_name = parsed["tool_call_json"].get("name")
                tool_args = parsed["tool_call_json"].get("arguments", {})

                # CHECK FOR ASYNC TOOL
                print(f"Checking if tool {tool_name} is async...")
                print(f"Async tools available: {[tool.__name__ for tool in self.async_tools]}")
                async_tool_names = [tool.__name__ for tool in self.async_tools]
                if tool_name in async_tool_names:
                    job_id = uuid.uuid4().hex
                    print(f"Async tool detected: {tool_name}. Suspending execution.")
                    
                    # 1. Save State
                    self.save_state(job_id, tool_name, tool_args)
                    
                    # 2. Notify User
                    yield previous_steps_parsed + [{
                        "content": f"Job {job_id} queued for {tool_name}. Shutting down to save resources.",
                        "tool_call_json": None,
                        "thinking": "",
                        "tool_result": None
                    }]
                    
                    # Wait for the yield to propagate to the frontend
                    time.sleep(2) 
                    
                    print("Exiting process for async tool execution.")
                    os._exit(0)
                    
                    # 4. Stop the loop strictly
                    return 

                # Normal Sync Tool Execution
                tool_result = self.run_tool_call(parsed)

                # Append tool result to messages
                self.messages.append({"role": "tool", "content": tool_result, "id": message_id})

                # TESTING: Add tool result as parsed content
                previous_steps_parsed.append({
                    "tool_result": tool_result,
                    "thinking": "",
                    "content": "",
                    "tool_call_json": None
                })

                # Remove user input for next step
                user_input = None

                # Decrement max steps
                max_steps -= 1
            else:
                return previous_steps_parsed

    def generate_step(self, user_input:str, message_id=None):
        if user_input and user_input.strip() != "":
            self.messages.append({"role": "user", "content": user_input, "id": message_id})

        # Sanitize thinking content from previous assistant messages
        # This prevents filling up the context with old thinking tags
        if self.sanitize_messages:
            prompt_input_messages = []
            for msg in self.messages:
                if msg["role"] == "assistant":
                    # Remove any existing <think>...</think> tags
                    content = msg.get("content", "")
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
                    msg["content"] = content.strip()
                prompt_input_messages.append(msg)
        else:
            prompt_input_messages = self.messages

        prompt = self.tokenizer.apply_chat_template(
            prompt_input_messages,
            tokenize=False,
            tools=self.tools + self.async_tools,
            add_generation_prompt=True,
        )

        # Log prompt for debugging
        os.makedirs("debug_logs", exist_ok=True)
        current_time = time.strftime("%Y%m%d_%H%M%S")
        with open(f"debug_logs/debug_prompt_{current_time}.txt", "w") as f:
            f.write(prompt)

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
            "max_new_tokens": int(4096*2), # Limit to 4096 tokens, for RAM saving, adjust as needed
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")],
            # Enable following in NVIDIA CUDA
            # If semantic repetition is an issue, enable these with adjusted params
            "do_sample": False,      # Enables sampling to break deterministic loops, proven buggy in apple silicon MPS
            "temperature": 0.5,     # Low temperature for more focused, less "creative" logic
                                    # Too low (0.3) can crash in apple silicon MPS
            #"repetition_penalty": 1.15, # 1.15 in nvidia CUDA, 1.0 in apple silicon MPS to avoid crashes
            "top_p": 0.95, # nucleus sampling
            "top_k": 50,   # top-k sampling
            
        }

        with self.model_lock, torch.no_grad():
            self.model.generate(**generation_kwargs)

        
        response = ""
        for new_text in streamer:
            response += new_text
            print("Generated so far:", response)
            parsed = self.parse_response(response, allow_tools=False)
            yield parsed

            #no yield response chunks for now, just return full response at the end
     
        # Clear thinking content from stored message
        self.messages.append({"role": "assistant", "content": response, "id": message_id})

        parsed = self.parse_response(response)
        #print("\n\n====== Parsed response:", parsed, "\n\n")
        # Log final response
        os.makedirs("debug_logs", exist_ok=True)
        with open(f"debug_logs/debug_response_{current_time}.txt", "w") as f:
            f.write(response)

        return parsed

    def parse_response(self, response: str, allow_tools=True) -> dict:
        # parsing thinking content and tool calls from the response
        thinking_token = "<think>"
        thinking_end_token = "</think>"
        thinking_content = ""

        tool_call_token = "<tool_call>"
        tool_call_end_token = "</tool_call>"
        tool_call_content = ""
        tool_call_json = None

        content = response

        if thinking_token in response:
            start_idx = response.index(thinking_token) + len(thinking_token)
            try:
                end_idx = response.index(thinking_end_token)
            except ValueError:
                # If no end tag, take till end of response
                # This happens during streaming before end tag is generated
                end_idx = len(response)

            thinking_content = response[start_idx:end_idx].strip()
            content = response[end_idx + len(thinking_end_token):].strip()

        if tool_call_token in response and allow_tools:
            start_idx = response.index(tool_call_token) + len(tool_call_token)
            try:
                end_idx = response.index(tool_call_end_token)
            except ValueError:
                # If no end tag, take till end of response
                # This happens during streaming before end tag is generated
                end_idx = len(response)

            tool_call_content = response[start_idx:end_idx].strip()
            content = content[end_idx + len(tool_call_end_token):].strip()
            print("Tool call content detected:", tool_call_content)
            try:
                tool_call_json = json.loads(tool_call_content)
            except json.JSONDecodeError:
                # This happens when streaming before full JSON is generated
                tool_call_json = None

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

            if tool_name in self.heavy_tools and self.model_loader:
                print(f"!!!! Unloading model for heavy tool: {tool_name}")
                # Unload model to free up memory
                if self.model_unloader:
                    # Unload model from framework
                    self.model_unloader()
            
            for tool in self.tools:
                print(f"Checking tool: {tool.__name__}")
                print(f"Looking for tool: {tool_name}")
                if tool.__name__ == tool_name:
                    try:
                        result = tool(**tool_args)
                        print(f"Tool {tool_name} executed with result: {result}")

                        if tool_name in self.heavy_tools and self.model_loader:
                            print(f"Reloading model after heavy tool: {tool_name}")
                            self.model = self.model_loader()

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
        
    def tool_result_processor(self, tool_name: str, result: str):
        # If there is <tool_result> tags, extract content
        # This prevents tools from verbose outputs overwhelming the model
        result_token = "<tool_result>"
        result_end_token = "</tool_result>"
        if result_token in result and result_end_token in result:
            start_idx = result.index(result_token) + len(result_token)
            end_idx = result.index(result_end_token)
            result_content = result[start_idx:end_idx].strip()
            return result_content
        
        return result

    def register_peer(self, name: str, agent_instance):
        """
        Registers another agent instance that this agent can communicate with.
        """
        self.peers[name] = agent_instance

    def talk_to_peer(self, peer_name: str, message: str) -> str:
        """
        Generic method to send a message to a registered peer agent and wait for the full response.
        This handles the complexity of consuming the other agent's generator.
        """
        if peer_name not in self.peers:
            available = list(self.peers.keys())
            return f"System Error: Agent '{peer_name}' is not known to me. Available peers: {available}"

        target_agent = self.peers[peer_name]
        print(f"\n[System] 🔄 {self.__class__.__name__} is delegating to {peer_name}...")

        # Create a unique ID for this sub-task
        task_id = uuid.uuid4().hex
        
        # Trigger the other agent's response loop
        # We pass the message and get back a generator (streaming response)
        response_generator = target_agent.respond(message, message_id=task_id)
        
        final_content = "No response generated."
        
        # We must consume the generator to let the other agent execute its tools
        for step in response_generator:
            # 'step' is a list of parsed chunks. We look at the latest state.
            last_state = step[-1]
            
            # Optional: Log tool usage from the peer for debugging
            if last_state.get("tool_call_json"):
                tool_name = last_state["tool_call_json"].get("name", "unknown")
                print(f"   ↳ [{peer_name}] Invoking tool: {tool_name}")
            
            # Update the final answer with the latest text content
            if last_state.get("content"):
                final_content = last_state["content"]

        print(f"[System] ✅ {peer_name} finished task.\n")
        return f"Response from {peer_name}:\n{final_content}"
    