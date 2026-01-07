import gradio as gr
import json
import uuid
import os
import argparse
import threading
import time

# Removed transformers imports
# from transformers import ... 

from framework.rag_engine import RAGEngine
from framework.orchestrator_agent import OrchestratorAgent
from framework.analytics_agent import AnalyticsAgent
from framework.logger import get_logger
# from framework.utilities.cuda_ram_debug import log_cuda_memory # Optional, likely not needed for Ollama

class Framework:
    def __init__(self, *args, **kwargs):
        # Default to a robust model available in Ollama
        self.base_model_name = kwargs.get('base_model_name')
        if not self.base_model_name:
            raise ValueError("Base model name must be provided for Ollama models.")
        
        # Initialize Logger
        self.logger = get_logger("Framework")
        
        # Initialize RAG Engine
        self.rag_engine = None  # Disable RAG for now

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
                        "3. ANALYZE: Read the report created by the Analyst.\n"
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
                        "1. SMART CACHING: Check if input/output files exist before running tools.\n"
                        "2. FULL PIPELINE: FastJet -> Preparation -> Training -> Oracle -> Report Generator.\n"
                        "3. PARTIAL RE-RUNS: You can re-run any step if parameters change or if directed by the Orchestrator.\n"
                    )
                }
            ],
        }

        # Initialize Agents with Model NAME, not object
        self.orchestrator_agent = OrchestratorAgent(
            model_name=self.base_model_name, # Changed arg name
            # tokenizer=self.tokenizer,      # Removed
            initial_messages=self.get_initial_messages(
                agent="OrchestratorAgent"
            ),
            rag_engine=self.rag_engine,
        )

        self.analytics_agent = AnalyticsAgent(
            model_name=self.base_model_name, # Changed arg name
            # tokenizer=self.tokenizer,      # Removed
            initial_messages=self.get_initial_messages(
                agent="AnalyticsAgent"
            ),
            rag_engine=self.rag_engine,
        )

        # Register peers
        self.orchestrator_agent.register_peer('AnalyticsAgent', self.analytics_agent)
        self.analytics_agent.register_peer('OrchestratorAgent', self.orchestrator_agent)

        # Register agents to framework
        self.agents = {
            "OrchestratorAgent": self.orchestrator_agent,
            "AnalyticsAgent": self.analytics_agent,
        }

        # If resume arg provided, trigger resume
        # Resume logic
        resume_job_id = kwargs.get('resume_job_id', None)

        if resume_job_id:
            self.logger.info(f"Resuming from job ID: {resume_job_id}")
            self.trigger_forced_resume(resume_job_id)
            self.forced_resume_in_progress = True
        else:
            self.forced_resume_in_progress = False
    

    def get_initial_messages(self, agent):
        return self.default_initial_messages.get(agent, [])

    # Removed load_model / unload_model functions

    def trigger_forced_resume(self, job_id):
        """Loads state from a completed job file to resume."""
        self.logger.info(f"Attempting to resume job {job_id}...")

        result_path = f"jobs/completed/{job_id}.json"

        if not os.path.exists(result_path):
            self.logger.info(f"Result file for job {job_id} not found at {result_path}. Cannot resume.")
            raise FileNotFoundError(f"Result file for job {job_id} not found.")

        with open(result_path, "r") as f:
            try:
                job_data = json.load(f)
            except json.JSONDecodeError as e:
                raise Exception(f"Error decoding JSON from {result_path}: {e}")
            except Exception as e:
                raise Exception(f"Error reading {result_path}: {e}")
        

        if (job_data and 
            job_data.get("original_state") and 
            job_data["original_state"].get("agent_identifier")
            ):
            agent_id = job_data["original_state"]["agent_identifier"]
            agent = self.agents.get(agent_id)
            if not agent:
                self.logger.info(f"Agent {agent_id} not found for resuming job {job_id}.")
                return
            self.logger.info(f"Resuming job {job_id} for agent {agent_id}.")

            agent.pending_job_id = job_id

            def run_generation():
                generator = agent.wait_for_tool_completion(job_id)
                for _ in generator:
                    pass  # Discard output during forced resume
                    # Poller will pick up the updated state

                self.forced_resume_in_progress = False
                self.logger.info(f"Resumed generation for job {job_id} completed.")

            threading.Thread(target=run_generation, daemon=True).start()
        else:
            self.logger.info(f"No valid original state found in job {job_id}. Cannot resume.")
            self.forced_resume_in_progress = False

    def check_background_updates(self):
        """Polled by Gradio Timer."""
        full_history = self.get_latest_history()
        
        if self.forced_resume_in_progress:
            return (
                full_history, 
                gr.update(interactive=False, placeholder="Resuming with previous job's results..."),
                gr.update(active=True) # Keep the timer running
            )
        else:
            return (
                full_history, 
                gr.update(interactive=True, placeholder="Type here..."), 
                gr.update(active=False) # Stop the timer
            )

    def chat_function(self, user_input, chat_history):
        message_id = uuid.uuid4().hex
        parsed_generator = self.analytics_agent.respond(user_input, message_id)

        for multiple_parsed in parsed_generator:
            full_history = self.analytics_agent.messages_to_gradio_history()
            
            # Transient bubbles
            current_bubbles = []
            for parsed in multiple_parsed:
                role = parsed.get("role", "assistant")
                if parsed["thinking"]:
                    current_bubbles.append(gr.ChatMessage(role=role, content=parsed["thinking"], metadata={"title": "Thinking"}))

                if parsed["tool_calls"]:
                    for tool_call in parsed["tool_calls"]:
                        current_bubbles.append(
                            gr.ChatMessage(
                                role=role, 
                                content=f"Invoking {tool_call['name']}...", 
                                metadata={"title": "Tool Call"})
                            )
                        
                if parsed["content"]:
                    current_bubbles.append(gr.ChatMessage(role=role, content=parsed["content"]))
            
            if current_bubbles:
                yield full_history + current_bubbles

    def get_latest_history(self):
        return self.analytics_agent.messages_to_gradio_history()
    
    def run_interactive(self):
        initial_gradio_history = self.get_latest_history()
        with gr.Blocks(fill_height=True) as demo:
            chatbot = gr.Chatbot(value=initial_gradio_history, type="messages", scale=1)
            msg = gr.Textbox(label="Your Input", placeholder="Type here...", autofocus=True)
            
            # Timer for background updates (Resume logic)
            if self.forced_resume_in_progress:
                interactive_polling_timer = gr.Timer(value=0.5, active=True)
                interactive_polling_timer.tick(fn=self.check_background_updates, inputs=None, outputs=[chatbot, msg, interactive_polling_timer])

            def submit_wrapper(user_input):
                last_history = [] 
                gen = self.chat_function(user_input, [])
                for history in gen:
                    last_history = history
                    yield gr.update(value="", interactive=False), history
                yield gr.update(interactive=True), last_history

            msg.submit(fn=submit_wrapper, inputs=[msg], outputs=[msg, chatbot])

        demo.launch(share=False)