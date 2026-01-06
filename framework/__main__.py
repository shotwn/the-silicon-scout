from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TextIteratorStreamer
import torch
import gradio as gr
import json
import uuid
import gc
import os
import argparse
import threading
import time

from framework.rag_engine import RAGEngine
from framework.orchestrator_agent import OrchestratorAgent
from framework.analytics_agent import AnalyticsAgent
from framework.utilities.cuda_ram_debug import log_cuda_memory

class Framework:
    def __init__(self, *args, **kwargs):
        # Flag to control BitsAndBytes usage
        allow_bnb = False

        # Determine device and dtype
        if torch.cuda.is_available():
            allow_bnb = True
            self.device = torch.device("cuda")
            self.dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float16
        
        # Disable BitsAndBytes on Mac entirely (it's slow and causes issues)
        if not allow_bnb:
            self.bnb_config = None
        else:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype 
            )
        
        self.base_model_name = kwargs.get('base_model_name', 'Qwen/Qwen3-4B')
        
        self.model = None
        self.load_model() # This sets self.model

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        # Initialize RAG Engine
        #self.rag_engine = RAGEngine()
        self.rag_engine = None  # Disable RAG for now

        # Trigger Ingestion (Only processes new files)
        #self.rag_engine.ingest_files("articles")

        # Initial messages per agent
        self.default_initial_messages = {
            "OrchestratorAgent": [
                {
                    "role": "system", "content": "You are an orchestrator agent part of a Particle Physics Anomaly Detection System. "
                    "Your task is to orchestrate different specialized agents to analyze data and provide insights."
                    "You use tool calls to interact with specialized agents as needed."
                    "To start, ask data file name from the user and wait for the response."
                    "After response, check if the file exists using the appropriate tool. Ask user if they want to proceed with data preprocessing using FastJet."
                }
            ],
            "AnalyticsAgent": [
                {
                    "role": "system", "content": "You are an analytics agent specialized in processing particle physics data files. "
                    "You have access to tools like LaCATHODE Trainer and Report Generator to analyze data files and generate reports. "
                    "Make sure to use tool calls in correct order to achieve the best results. "
                    "You can run training and report part multiple times with different parameters and merge or compare the results into a final report. "
                    "But make sure to not overwrite previous runs unless specified. "
                    "\nA usual run would be: "
                    "1. fastjet_tool "
                    "2. lacathode_preperation_tool "
                    "3. lacathode_trainer_tool "
                    "4. lacathode_oracle_tool "
                    "5. lacathode_report_generator_tool "
                    "\nIMPORTANT: Use list_any_cwd_folder tool to check for files and directories in current working directory when needed. "
                    "\nIMPORTANT: When you are not sure about something, ask the user for clarification before diving in long thinking or tool calls."
                    "\nIMPORTANT: Focus ONLY on the immediate next step. Do not try to plan or validate arguments for future tools (like step 2 or 3) "
                    "until the current tool execution is complete and you have the results. "
                    "Execute the first necessary tool immediately."
                }
            ],
        }

        # Resume logic
        resume_job_id = kwargs.get('resume_job_id', None)
        resume_job_data = self.get_resume_job_data(resume_job_id)
        """
        self.orchestrator_agent = OrchestratorAgent(
            model=self.model,
            tokenizer=self.tokenizer,
            initial_messages=self.get_initial_messages(
                agent="OrchestratorAgent",
                resume_job_data=resume_job_data
            ),
            rag_engine=self.rag_engine,
            model_loader=self.load_model,
            model_unloader=self.unload_model
        )
        """
        self.analytics_agent = AnalyticsAgent(
            model=self.model,
            tokenizer=self.tokenizer,
            initial_messages=self.get_initial_messages(
                agent="AnalyticsAgent",
                resume_job_data=resume_job_data
            ),
            rag_engine=self.rag_engine,
            model_loader=self.load_model,
            model_unloader=self.unload_model
        )

        # If resuming, process the last tool result offline in a separate thread
        # This ensures Gradio launches immediately while the agent works in the background
        resume_thread = threading.Thread(target=self.process_offline_resume)
        resume_thread.daemon = True # Ensures thread cleans up when main process exits
        resume_thread.start()
    
    def get_resume_job_data(self, resume_job_id: str):
        if resume_job_id:
            print(f"Resuming session from Job ID: {resume_job_id}")
            result_path = f"jobs/completed/{resume_job_id}.json"
            
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    data = json.load(f)
                
                return data
            else:
                print("Job result file not found!")
        
        return None

    def get_initial_messages(self, agent, resume_job_data=None):
        if (
            resume_job_data and 
            resume_job_data.get("original_state") and 
            resume_job_data["original_state"].get("agent_identifier") == agent
        ):
            # Resume from previous state
            print(f"Restoring messages for {agent} from resume data.")
            initial_messages = resume_job_data["original_state"]["messages"]

            # Inject tool result from previous run
            tool_result_msg = {
                "role": "tool",
                "content": resume_job_data["tool_result"],
            }

            initial_messages.append(tool_result_msg)

            return initial_messages
        else:
            # Fresh start
            return self.default_initial_messages.get(agent, [])
    
    def process_offline_resume(self):
        # Wait a moment to ensure Gradio has launched
        time.sleep(2)
        # Check if the last message in history is a Tool Result
        if self.analytics_agent.messages and self.analytics_agent.messages[-1]['role'] == 'tool':
            print(">>> Auto-Resume detected: Processing Tool Result offline...")
            
            # Run the agent loop with empty input to process the result
            # We iterate over the generator to force execution, but ignore the output
            generator = self.analytics_agent.respond("", message_id=uuid.uuid4().hex)
            try:
                for _ in generator:
                    pass
                print(">>> Offline processing complete.")
            except SystemExit:
                pass # Handle async tool exits gracefully

    def load_model(self):
        print("Framework: Loading model reference...")
        log_cuda_memory("CUDA Memory Before loading model")
        max_memory = None
        device_map = None
        if torch.cuda.is_available():
            max_memory = {0: "7GB"}
            device_map = "auto"
        elif torch.backends.mps.is_available():
            max_memory = {"mps": "14GB"}
            device_map = {"": torch.device("mps")}
        
        
        # Check if we are on Mac/MPS to determine config
        if torch.backends.mps.is_available():
            quantization_config = None
            #attn_implementation = "sdpa"  # sdpa is faster but can cause inf errors 
            attn_implementation = "eager"
        else:
            quantization_config = self.bnb_config
            attn_implementation = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map=device_map,
            max_memory=max_memory,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
            torch_dtype=self.dtype
        )
        log_cuda_memory("CUDA Memory After loading model")

        self.model.eval()

        return self.model
    
    def unload_model(self):
        if self.model is not None:
            print("Framework: Unloading model reference...")
            log_cuda_memory("CUDA Memory Before unloading model for heavy tool")
            # Move to CPU first to be safe (optional but helps detach hooks)
            try:
                self.model.to("cpu") 
            except Exception:
                pass

            del self.model
            self.model = None
            torch.cuda.empty_cache()

            for obj in gc.get_objects():
                # Safely attempt to delete CUDA tensors
                try:
                    if isinstance(obj, torch.Tensor) and obj.is_cuda:
                        del obj
                except Exception:
                    continue

            gc.collect()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            
            torch.cuda.empty_cache()

            log_cuda_memory("CUDA Memory After unloading model for heavy tool")
    
    def chat_function(self, user_input, chat_history):
        message_id = uuid.uuid4().hex

        # Create the user message object manually to append to history
        user_msg = gr.ChatMessage(role="user", content=user_input)

        # Get the generator from the agent
        parsed_generator = self.analytics_agent.respond(user_input, message_id, chat_history)

        for multiple_parsed in parsed_generator:
            response_bubbles = []
            for parsed in multiple_parsed:
                if parsed["thinking"]:
                    response_bubbles.append(gr.ChatMessage(
                        role="assistant",
                        content=f"{parsed['thinking']}",
                        metadata={"title": "Thinking", "id": message_id}
                    ))

                if parsed["tool_call_json"]:
                    tool_name = parsed["tool_call_json"].get("name", "unknown_tool")

                    response_bubbles.append(gr.ChatMessage(
                        role="assistant",
                        content=f"Invoking tool... ({tool_name})",
                        metadata={"title": "Tool Call", "id": message_id}
                    ))

                if parsed.get("tool_result"):
                    response_bubbles.append(gr.ChatMessage(
                        role="assistant", 
                        content=f"Tool Result:\n{parsed['tool_result']}",
                        metadata={"title": "Tool Result", "id": message_id}
                    ))
                
                if parsed["content"]:
                    response_bubbles.append(gr.ChatMessage(
                        role="assistant", 
                        content=parsed["content"],
                        metadata={"id": message_id}
                    ))
            
            if response_bubbles:
                yield chat_history + [user_msg] + response_bubbles

    def get_latest_history(self):
        """Helper to fetch the current agent state for the UI"""
        return self.analytics_agent.messages_to_gradio_history()
    
    def run_interactive(self):
        initial_gradio_history = self.get_latest_history()
        with gr.Blocks(fill_height=True) as demo:
            chatbot = gr.Chatbot(
                value=initial_gradio_history,
                type="messages",
                scale=1
            )

            chatbot.clear(fn=lambda: self.analytics_agent.messages.clear())

            gr.ChatInterface(
                fn=self.chat_function,
                title="Interactive Chat with Analytics Agent",
                description="Chat interface for interacting with the analytics agent.",
                type="messages",
                chatbot=chatbot,
            )

            # AUTO-RESUME TRIGGER
            # If we are resuming, we want the bot to look at the Tool Result 
            # (which is the last message) and generate an answer immediately.
            """
            if self.analytics_agent.messages and self.analytics_agent.messages[-1]['role'] == 'tool':
                def auto_trigger():
                    # Pass an empty user string, the agent handles the rest
                    yield from self.chat_function("")
                
                # Queue this to run immediately on load
                demo.load(auto_trigger, outputs=[chatbot])
            """
            demo.load(fn=self.get_latest_history, outputs=chatbot)
        demo.launch(share=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Job ID to resume from")
    args = parser.parse_args()

    framework = Framework(base_model_name='Qwen/Qwen3-4B', resume_job_id=args.resume)
    framework.run_interactive()

    """
    chat1 = "events_anomalydetection_tiny.h5 go ahead and run this without checking if it exists"
    chat2 = "hi"
    gen = framework.chat_function(chat1, [])
    from framework.utilities.console_utilities import set_cursor_position, clear_screen
    clear_screen()
    for response in gen:
        # Every generator yield contains multiple parsed steps
        set_cursor_position(0,0)

        for message in response:
            if message.metadata and "title" in message.metadata:
                print(f"{message.role} ({message.metadata['title']}): {message.content}\n")
            else:
                print(f"{message.role}: {message.content}\n")

    # Wait for user input to exit
    input("Press Enter to exit...\n")
    """

    

