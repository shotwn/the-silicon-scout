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
            allow_bnb = True
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
                    "role": "system", 
                    "content": (
                        "You are a Senior Particle Physicist (The Scientist). "
                        "You direct an Analyst Agent to find anomalies in collider data. "
                        "Your Goal: Discover new physics anomalies with >3 sigma significance.\n\n"
                        "PROTOCOL - THE REPORTING LOOP:\n"
                        "1. DIRECT: Issue high-level commands to the Analyst (e.g., 'Run analysis on the 3.5 TeV region').\n"
                        "2. READ: When the Analyst tells you a report has been generated (e.g., 'llm_enhanced_report.txt'), "
                        "you MUST use your file reading tool (e.g., 'read_any_cwd_file' or 'read_article_file') to inspect the contents of that file immediately.\n"
                        "3. ANALYZE: Look for 'Max Excess' and 'Significance' in the file content you just read.\n"
                        "4. DECIDE: \n"
                        "   - If Significance > 3.0: Recommend publication.\n"
                        "   - If Significance < 3.0: Formulate a new hypothesis (e.g., 'The signal might be softer, let's lower min_pt') and command the Analyst to re-run.\n"
                        "   - If Significance is high near band edges: Suggest refining the mass range.\n"
                        "5. DONE: Only when you have a strong discovery or have exhausted options, start your response with 'FINAL REPORT'."
                    )
                }
            ],
            "AnalyticsAgent": [
                {
                    "role": "system", 
                    "content": (
                        "You are an Expert Research Technician (The Analyst). "
                        "You operate the LaCATHODE anomaly detection pipeline. "
                        "You have access to tools: FastJet, Preparation, Trainer, Oracle, and Report Generator.\n\n"
                        "EXECUTION RULES:\n"
                        "1. SMART CACHING: Before running 'fastjet_tool', ALWAYS check if the output files (e.g., 'signal_events.jsonl') already exist using 'list_any_cwd_folder'. "
                        "If they exist, SKIP Step 1. Do not overwrite existing data unless explicitly asked.\n"
                        "2. PIPELINE: FastJet -> Preparation -> Training -> Oracle -> Report Generator.\n"
                        "3. HANDOFF: The 'lacathode_report_generator_tool' will output a filename (e.g., 'llm_enhanced_report.txt') but NOT the content. "
                        "Your job is to generate this file and then immediately tell the Scientist: 'Report generated at [filename]. Please read it for details.' "
                        "Do NOT try to summarize the report yourself, as you cannot see the content."
                    )
                }
            ],
        }

        # Resume logic
        resume_job_id = kwargs.get('resume_job_id', None)
        resume_job_data = self.get_resume_job_data(resume_job_id)

        self.orchestrator_agent = OrchestratorAgent(
            model=self.model,
            tokenizer=self.tokenizer,
            initial_messages=self.get_initial_messages(
                agent="OrchestratorAgent",
                resume_job_data=resume_job_data
            ),
            rag_engine=self.rag_engine,
        )

        self.analytics_agent = AnalyticsAgent(
            model=self.model,
            tokenizer=self.tokenizer,
            initial_messages=self.get_initial_messages(
                agent="AnalyticsAgent",
                resume_job_data=resume_job_data
            ),
            rag_engine=self.rag_engine,
        )

        # Register peers
        self.orchestrator_agent.register_peer('AnalyticsAgent', self.analytics_agent)
        self.analytics_agent.register_peer('OrchestratorAgent', self.orchestrator_agent)

        # If resuming, process the last tool result offline in a separate thread
        # This ensures Gradio launches immediately while the agent works in the background
        self.offline_resume_temp_data = None
        """
        resume_thread = threading.Thread(target=self.process_offline_resume)
        resume_thread.daemon = True # Ensures thread cleans up when main process exits
        resume_thread.start()
        """
        self.process_offline_resume()
    
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

            # Extract the last toold request's id
            last_tool_call = None
            for msg in reversed(initial_messages):
                if msg['role'] == 'assistant' and 'tool_call' in msg:
                    last_tool_call = msg
                    break

            tool_call_id = None
            if last_tool_call:
                print("No tool call found in previous messages!")
                tool_call_id = last_tool_call.get('id', None)
            
            if not tool_call_id:
                tool_call_id = uuid.uuid4().hex

            # Inject tool result from previous run
            user_msg = {
                "role": "user",
                "content": "Resuming from previous tool result.",
                "id": tool_call_id
            }
            tool_result_msg = {
                "role": "tool",
                "content": resume_job_data["tool_result"],
                "id": tool_call_id
            }

            #initial_messages.append(user_msg)
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
            generator = self.analytics_agent.respond("", message_id=self.analytics_agent.messages[-1]['id'])
            try:
                for parsed in generator:
                    #self.offline_resume_temp_data = parsed
                    print("parsed: " + str(parsed))
                print(">>> Offline processing complete.")
            except SystemExit:
                pass # Handle async tool exits gracefully
            finally:
                self.offline_resume_temp_data = None

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
        """Gradio chat function to interact with the analytics agent
        Yields updated chat history with streaming responses.

        Args:
            user_input (str): The user's input message.
            chat_history (list): The current chat history. Not used, as we fetch from agent state.
        """
        message_id = uuid.uuid4().hex

        # Get the generator from the agent
        parsed_generator = self.analytics_agent.respond(user_input, message_id)

        for multiple_parsed in parsed_generator:
            # Keep it in sync with agents memory
            full_history = self.analytics_agent.messages_to_gradio_history()
            
            # In progress response bubbles
            current_response_bubbles = []
            for parsed in multiple_parsed:
                if parsed["thinking"]:
                    current_response_bubbles.append(gr.ChatMessage(
                        role="assistant",
                        content=f"{parsed['thinking']}",
                        metadata={"title": "Thinking", "id": message_id}
                    ))

                if parsed["tool_call_json"]:
                    tool_name = parsed["tool_call_json"].get("name", "unknown_tool")

                    current_response_bubbles.append(gr.ChatMessage(
                        role="assistant",
                        content=f"Invoking tool... ({tool_name})",
                        metadata={"title": "Tool Call", "id": message_id}
                    ))

                if parsed.get("tool_result"):
                    current_response_bubbles.append(gr.ChatMessage(
                        role="assistant", 
                        content=f"Tool Result:\n{parsed['tool_result']}",
                        metadata={"title": "Tool Result", "id": message_id}
                    ))
                
                if parsed["content"]:
                    current_response_bubbles.append(gr.ChatMessage(
                        role="assistant", 
                        content=parsed["content"],
                        metadata={"id": message_id}
                    ))
            
            if current_response_bubbles: # Only yield if there is something new
                yield full_history + current_response_bubbles

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

            msg = gr.Textbox(label="Your Input", placeholder="Type here...", autofocus=True)

            def submit_wrapper(user_input):
                # Initialize backup history in case generator fails immediately
                last_history = [] 
                
                gen = self.chat_function(user_input, [])
                
                # STREAMING: Lock input box, clear text, update chat
                for history in gen:
                    last_history = history
                    # interactive=False disables the box while processing
                    yield gr.update(value="", interactive=False), history
                
                # FINISHED: Unlock input box, keep final chat history
                yield gr.update(interactive=True), last_history

            msg.submit(fn=submit_wrapper, inputs=[msg], outputs=[msg, chatbot])

            #demo.load(fn=self.get_latest_history, outputs=chatbot)
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

    

