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
# from framework.utilities.cuda_ram_debug import log_cuda_memory # Optional, likely not needed for Ollama

class Framework:
    def __init__(self, *args, **kwargs):
        # Default to a robust model available in Ollama
        self.base_model_name = kwargs.get('base_model_name')
        if not self.base_model_name:
            raise ValueError("Base model name must be provided for Ollama models.")
        
        # We no longer load the model into Python memory
        # self.model = None 
        
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

        # Resume logic
        resume_job_id = kwargs.get('resume_job_id', None)
        resume_job_data = self.get_resume_job_data(resume_job_id)

        # Initialize Agents with Model NAME, not object
        self.orchestrator_agent = OrchestratorAgent(
            model_name=self.base_model_name, # Changed arg name
            # tokenizer=self.tokenizer,      # Removed
            initial_messages=self.get_initial_messages(
                agent="OrchestratorAgent",
                resume_job_data=resume_job_data
            ),
            rag_engine=self.rag_engine,
        )

        self.analytics_agent = AnalyticsAgent(
            model_name=self.base_model_name, # Changed arg name
            # tokenizer=self.tokenizer,      # Removed
            initial_messages=self.get_initial_messages(
                agent="AnalyticsAgent",
                resume_job_data=resume_job_data
            ),
            rag_engine=self.rag_engine,
        )

        # Register peers
        self.orchestrator_agent.register_peer('AnalyticsAgent', self.analytics_agent)
        self.analytics_agent.register_peer('OrchestratorAgent', self.orchestrator_agent)

        # Resume processing
        self.offline_resume_temp_data = None
        
        # Start resume thread
        resume_thread = threading.Thread(target=self.process_offline_resume)
        resume_thread.daemon = True 
        resume_thread.start()
    
    def get_resume_job_data(self, resume_job_id: str):
        if resume_job_id:
            print(f"Resuming session from Job ID: {resume_job_id}")
            result_path = f"jobs/completed/{resume_job_id}.json"
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    return json.load(f)
            else:
                print("Job result file not found!")
        return None

    def get_initial_messages(self, agent, resume_job_data=None):
        if (
            resume_job_data and 
            resume_job_data.get("original_state") and 
            resume_job_data["original_state"].get("agent_identifier") == agent
        ):
            print(f"Restoring messages for {agent} from resume data.")
            initial_messages = resume_job_data["original_state"]["messages"]
            
            # Simple resume logic: inject tool result
            tool_call_id = uuid.uuid4().hex

            # Get the last tool call made by this agent
            last_tool_call_name = resume_job_data["original_state"].get("tool_name", None)
            
            # Note: We need to append the result to conversation
            # The last message in 'messages' should be the assistant's tool call
            # We append the result as a 'tool' role message
            
            tool_result_msg = {
                "role": "tool",
                "content": resume_job_data["tool_result"],
                "id": tool_call_id,
                "tool_name": last_tool_call_name,
            }
            initial_messages.append(tool_result_msg)
            return initial_messages
        else:
            return self.default_initial_messages.get(agent, [])
    
    def process_offline_resume(self):
        time.sleep(2)
        if self.analytics_agent.messages and self.analytics_agent.messages[-1]['role'] == 'tool':
            print(">>> Auto-Resume detected: Processing Tool Result offline...")
            # Run the agent loop with empty input
            generator = self.analytics_agent.respond("", message_id=uuid.uuid4().hex)
            try:
                for parsed in generator:
                    self.offline_resume_temp_data = parsed
                print(">>> Offline processing complete.")
            except SystemExit:
                pass 
            finally:
                self.offline_resume_temp_data = None

    # Removed load_model / unload_model functions

    def check_background_updates(self):
        """Polled by Gradio Timer."""
        full_history = self.get_latest_history()
        
        if self.offline_resume_temp_data:
            # Manually construct bubble from temp data
            parsed = self.offline_resume_temp_data
            if isinstance(parsed, list): parsed = parsed[-1] # handle list wrap
            
            if parsed.get("thinking"):
                full_history.append(gr.ChatMessage(role="assistant", content=parsed["thinking"], metadata={"title": "Auto-Thinking"}))

            if parsed.get("tool_calls"):
                for tool_call in parsed["tool_calls"]:
                    full_history.append(
                        gr.ChatMessage(
                            role="assistant", 
                            content=f"Invoking {tool_call['name']}...", 
                            metadata={"title": "Auto-Tool Call"})
                        )
            if parsed.get("content"):
                full_history.append(gr.ChatMessage(role="assistant", content=parsed["content"]))
                
            return full_history, gr.update(interactive=False, placeholder="Agent is auto-resuming...")
        
        return full_history, gr.update(interactive=True, placeholder="Type here...")

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
            timer = gr.Timer(value=0.5, active=True)
            timer.tick(fn=self.check_background_updates, inputs=None, outputs=[chatbot, msg])

            def submit_wrapper(user_input):
                last_history = [] 
                gen = self.chat_function(user_input, [])
                for history in gen:
                    last_history = history
                    yield gr.update(value="", interactive=False), history
                yield gr.update(interactive=True), last_history

            msg.submit(fn=submit_wrapper, inputs=[msg], outputs=[msg, chatbot])

        demo.launch(share=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Job ID to resume from")
    parser.add_argument("--model", type=str, default="qwen3:14b", help="Ollama model name")
    args = parser.parse_args()

    framework = Framework(base_model_name=args.model, resume_job_id=args.resume)
    framework.run_interactive()