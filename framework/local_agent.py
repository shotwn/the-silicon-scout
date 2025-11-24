"""
LocalAgent class for handling local LLM interactions with tool usage.
This is parent class for specialized agents like OrchestratorAgent.
"""
import threading
import json
import gc
import torch

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

        self.heavy_tools = []  # To be populated with get_tools or manually
        self.tools = self.get_tools()
    
    def get_tools(self):
        # Define specialized agents/tools here
        return []
    
    def respond(self, user_input:str, message_id=None, chat_history=None) -> list:
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
                # Run the tool call
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

        prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            tools=self.tools,
            add_generation_prompt=True,
        )

        # Print the final prompt for debugging
        print("\n\nFinal prompt to model:")
        print(prompt, "\n\n")

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
            "max_new_tokens": int(32768 / 8), # Limit to 4k tokens, for RAM saving, adjust as needed
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
            parsed = self.parse_response(response)
            yield parsed

            #no yield response chunks for now, just return full response at the end
     

        self.messages.append({"role": "assistant", "content": response, "id": message_id})

        parsed = self.parse_response(response)
        print("\n\n====== Parsed response:", parsed, "\n\n")

        # Make sure thread has finished
        thread.join()

        # Clear memory, in case there will be subsequent tool calls
        try:
            del generation_kwargs
            del thread
            del inputs
            del input_ids
            del attention_mask
            del streamer
            del generate
        except Exception as e:
            print(e)
            pass
        
        # Thorough cleanup
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

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
    