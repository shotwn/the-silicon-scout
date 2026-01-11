import gradio as gr
import json
import uuid
import os
import argparse
import threading
import time
import glob
from dotenv import load_dotenv

# Removed transformers imports
# from transformers import ... 

from framework.orchestrator_agent import OrchestratorAgent
from framework.analytics_agent import AnalyticsAgent
from framework.tools.gemma_client import get_runtime_history_file
from framework.logger import get_logger
from logging import DEBUG, INFO, WARNING, ERROR
# from framework.utilities.cuda_ram_debug import log_cuda_memory # Optional, likely not needed for Ollama

# Load environment variables from .env file immediately
load_dotenv()

class Framework:
    def __init__(self, *args, **kwargs):
        # Default to a robust model available in Ollama
        self.base_model_name = kwargs.get('base_model_name')
        if not self.base_model_name:
            raise ValueError("Base model name must be provided for Ollama models.")
        
        self.session_id = self._initialize_session()
        
        # Initialize Logger
        self.logger = get_logger("Framework", level=INFO)
        
        # Initialize RAG Engine
        self.rag_engine_enabled = True

        # Initial messages per agent
        self.default_initial_messages = {
            "OrchestratorAgent": [
                {
                    "role": "system", 
                    "content": (
                        "You are a Senior Particle Physicist (The Scientist). "
                        "You direct an Analyst Agent to uncover hidden physics in collider data. "
                        "Your Goal: Make a verified discovery of New Physics (>2-4 sigma) or rigorously exclude it.\n\n"
                        
                        "## YOUR OPERATIONAL PHILOSOPHY\n"
                        "You are not a script runner; you are a Principal Investigator. You are expected to:\n"
                        "1. **Think Strategically:** Don't just run tools blindly. Formulate hypotheses based on the data structure.\n"
                        "2. **Be Skeptical:** A bump in the data might be a signal, or it might be a detector artifact. Verify before celebrating.\n"
                        "3. **Be Efficient:** Training models is expensive. Always validate your parameters (windows, bins) before committing to a full run.\n"
                        "4. **DO NOT Micromanage File Formats:** Never specify output formats like '.root', '.csv', or '.h5'. "
                        "   The Analyst's tools have hard-coded internal formats (JSONL/NPY). Trust the Analyst to handle I/O.\n"

                        "## CORE SCIENTIFIC WORKFLOW (Flexible Guide)\n"
                        "While you have creative freedom in your approach, you must generally adhere to this scientific logic:\n\n"
                        
                        "### A. RECONNAISSANCE (Know Your Data)\n"
                        "   - Never assume file contents. Use your file listing tools to survey the directory first.\n"
                        "   - Ensure raw data is processed (FastJet). If files are missing, command the Analyst to generate them.\n\n"
                        
                        "### B. STRATEGIC PLANNING (The Feasibility Check)\n"
                        "   - **CRITICAL STEP:** Before searching for anomalies, you must define *where* to search.\n"
                        "   - Command the Analyst to 'Propose Signal Regions'.\n"
                        "   - **WARNING:** The 'Propose' tool does NOT find physics. It calculates mathematical feasibility (Sideband Statistics). "
                        "     It tells you: 'You *can* run the algorithm here without crashing.' It does *not* tell you: 'There is a particle here.'\n"
                        "   - Select a region that balances high statistics with physical interest.\n\n"
                        
                        "### C. EXECUTION (The Hunt)\n"
                        "   - Command the Analyst to run the specific anomaly detection pipeline on your chosen window.\n"
                        "   - You can deviate from standard parameters if you have a specific physics intuition (e.g., specific mass ranges), "
                        "     but verify they are safe first.\n\n"
                        
                        "### D. ANALYSIS & DEDUCTION\n"
                        "   - When a result file is produced, read it IMMEDIATELY. Do not wait.\n"
                        "   - Use your `query_knowledge_base_tool` to interpret findings (e.g., 'What particles decay into 2-jet resonances at 3 TeV?').\n"
                        "   - Use `query_gemma_cloud_tool` for broad literature checks or if you are stuck.\n\n"
                        
                        "## IMPORTANT LIMITATIONS (Non-Negotiable)\n"
                        "1. **DATA INTEGRITY:** You may have intuition, but you must NEVER invent data. "
                        "   All numbers (p-values, masses, event counts) must come directly from the Analyst's reports.\n"
                        "2. **TOOL REALITY:** You cannot 'look' at plots directly. You must read text files (.json, .txt, .csv) generated by the tools.\n"
                        "3. **DELEGATION:** You command the Analyst. Be clear, precise, and unambiguous with paths and parameter names. "
                        "   Do not micromanage *how* they run code, but be strict about *what* parameters they use.\n"
                        "4. **SAFETY FIRST:** Never run a training loop on a window without confirming it has sufficient sidebands (via the 'Propose' step). "
                        "   Blind runs will fail and waste time.\n\n"

                        "## FINAL DELIVERABLE\n"
                        "Only when you have a strong result or have exhausted all reasonable searches, start your response with 'FINAL REPORT'. "
                        "Your report must include:\n"
                        "1. **Null Hypothesis p-value:** The probability the data contains no new physics.\n"
                        "2. **Characterization:** The mass and decay modes of the new particle (if found) with uncertainties.\n"
                        "3. **Signal Strength:** Number of signal events extracted (+/- uncertainty).\n\n"
                        
                        "This is session ID: " + self.session_id + "\n"
                    )
                }
            ],
            "AnalyticsAgent": [
                {
                    "role": "system", 
                    "content": (
                        "You are an Expert Research Technician (The Analyst). "
                        "Your user is an another LLM agent called the Orchestrator (The Scientist). "
                        "You are capable of operating the LaCATHODE anomaly detection pipeline, Signal Region Recommendations and Isolation Forest. "
                        "Stick to physics, use units properly, and avoid making up numbers.\n"
                        "Do not ask tool specific questions to the Orchestrator, you have full autonomy to run tools as needed.\n\n"
                        "If instructed, give information about your tools and parameters.\n\n"
                        "## EXECUTION RULES:\n"
                        "1. FILE EXISTENCE: Check if input/output files exist with list_any_folder tool before running tools.\n"
                        "2. NO OVERWRITES: Do not overwrite existing files, unless explicitly instructed by the Orchestrator.\n"
                        "3. NO DUPLICATE RUN IDS: Always use a new run_id for each LaCATHODE tool invocation to avoid conflicts.\n"
                        "4. TOOL USAGE: One tool at a time, wait for completion before next.\n"
                        "5. PIPELINES:" 
                        "5.1. FastJet -> Signal Region Proposal -> Preparation -> Training -> Oracle -> Report Generator.\n"
                        "5.2. You can run Isolation Forest as an independent check after FastJet and if needed Signal Region Proposal.\n"
                        "6. WHEN TO REPORT: If ONLY a specific step is asked, report IMMEDIATELY after that step completes."
                        "If full pipeline is run, report AFTER the Report Generator step completes.\n"
                        "7. PARTIAL RE-RUNS: You can re-run steps to increase report data, but take cost into account.\n"
                        "8. REPORTING: After generating a report, inform the Orchestrator and await further instructions.\n\n"
                        "## EXECUTION RULES:\n"
                        "1. CHECKPOINTS: You are encouraged to STOP and report back after 'FastJet' and 'Region Proposal'. \n"
                        "2. FILE SAFETY: Check file existence before reading. Do not overwrite unless instructed.\n"
                        "3. RUN IDs: If not received by orchestrator, generate unique run_ids for every new training attempt.\n"
                        "4. AUTONOMY: If the Orchestrator says 'Find anomalies' without specifics, you MAY run the full pipeline autonomously.\n\n"
                        "5. EASY WITH PYTHON: You have access to a Python REPL tool. Use it for calculations, file inspections, or data manipulations. \n"
                        "   But make sure to not create workloads that will take hours to run, rely on existing tools first. Keep it efficient and quick.\n\n"
                        "## EXAMPLE-RERUNS:\n"
                        "- If the Orchestrator suggests lowering min_pt, you can re-run Preparation and subsequent steps.\n"
                        "- If the Orchestrator wants a finer mass binning, you can re-run Report Generator with updated parameters.\n"
                        "- If you are not satisfied with the report, you can re-run the report generator\n\n"
                        "Use your own judgement to balance cost vs information gain when re-running tools. \n"
                        f"   This is session ID: {self.session_id}\n"
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
            rag_engine_enabled=self.rag_engine_enabled,
        )

        self.analytics_agent = AnalyticsAgent(
            model_name=self.base_model_name, # Changed arg name
            # tokenizer=self.tokenizer,      # Removed
            initial_messages=self.get_initial_messages(
                agent="AnalyticsAgent"
            ),
            rag_engine_enabled=self.rag_engine_enabled,
        )

        # Register peers
        self.orchestrator_agent.register_peer('AnalyticsAgent', self.analytics_agent)
        self.analytics_agent.register_peer('OrchestratorAgent', self.orchestrator_agent)

        # Register agents to framework
        self.agents = {
            "OrchestratorAgent": self.orchestrator_agent,
            "AnalyticsAgent": self.analytics_agent,
        }

        # Make one active agent
        self.active_agent = self.orchestrator_agent

        # If resume arg provided, trigger resume
        # Resume logic
        resume_job_id = kwargs.get('resume_job_id', None)

        if resume_job_id:
            self.logger.info(f"Resuming from job ID: {resume_job_id}")
            self.trigger_forced_resume(resume_job_id)
            self.forced_resume_in_progress = True
        else:
            self.forced_resume_in_progress = False

        self._last_served_histories = {}

        self.chat_lock = threading.Lock()

        self.is_processing = False
        self.placeholder_text = "Type here..."
    

    def _initialize_session(self):
        counter_path = "session_counter.txt"
        os.makedirs("jobs", exist_ok=True)

        # Read the last number safely
        if os.path.exists(counter_path):
            with open(counter_path, "r") as f:
                try:
                    count = int(f.read().strip())
                except ValueError:
                    count = 0
        else:
            count = 0

        # Increment and save immediately to "lock" this session number
        new_count = count + 1
        with open(counter_path, "w") as f:
            f.write(str(new_count))

        device_tag = os.environ.get("DEVICE_TAG")
        if device_tag:
            session_id = f"{device_tag}_{new_count:03d}"
        else:
            session_id = f"SESS_{new_count:03d}"

        os.environ["FRAMEWORK_SESSION_ID"] = session_id

        return session_id

    def get_initial_messages(self, agent):
        return self.default_initial_messages.get(agent, [])

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

    def check_background_updates(self, agent_name=None):
        """Polled by Gradio Timer."""
        full_history = self.get_agent_history(agent_name)
        last_served_history = self._last_served_histories.get(agent_name, [])

        # Determine the visual "Tell"
        if self.is_processing:
            placeholder_text = "⌛ Orchestrator is thinking..."
            interactive_state = False # Block input while busy
        elif self.forced_resume_in_progress:
            placeholder_text = "Resuming..."
            interactive_state = False
        else:
            placeholder_text = "Type here..."
            interactive_state = True

        # If placeholder and history are unchanged and processing is not active, skip update
        if full_history == last_served_history and placeholder_text == self.placeholder_text and not self.is_processing:
                return (gr.skip(), gr.skip())
        
        self._last_served_histories[agent_name] = full_history
        self.placeholder_text = placeholder_text
        
        return (full_history, gr.update(interactive=interactive_state, placeholder=placeholder_text))

    def export_all_histories(self):
        """
        Dumps the full message history of all agents to a JSON file for download.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("exports", f"session_history_{self.session_id}.{timestamp}.json")
        os.makedirs("exports", exist_ok=True)
        
        # Collect data
        export_data = {
            "framework_version": "1.0",  # Placeholder version
            "base_model": self.base_model_name,
            "rag_engine_enabled": self.rag_engine_enabled,
            "session_id": self.session_id,
            "timestamp": timestamp,
            "agents": {}
        }
        
        for name, agent in self.agents.items():
            export_data["agents"][name] = agent.messages
            
        # Write to file
        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)
            
        return filename
    
    def import_all_histories(self, filepath):
        """
        Loads message history from a JSON file and restores agent states.
        """
        if not filepath:
            return "No file provided."

        self.logger.info(f"Attempting to import session from {filepath}")

        try:
            with open(filepath, "r", encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate format
            if "agents" not in data:
                return "Error: Invalid session file format (missing 'agents' key)."

            # Restore history for each agent found in the file
            loaded_count = 0
            for agent_name, messages in data["agents"].items():
                if agent_name in self.agents:
                    self.agents[agent_name].messages = messages
                    loaded_count += 1
                    self.logger.info(f"Restored {len(messages)} messages for {agent_name}.")
                else:
                    self.logger.warning(f"Agent '{agent_name}' found in file but not in current framework. Skipping.")
            
            return f"Successfully restored session (Timestamp: {data.get('timestamp', 'unknown')}). Loaded {loaded_count} agents."

        except json.JSONDecodeError:
            return "Error: Failed to decode JSON file."
        except Exception as e:
            self.logger.error(f"Import failed: {str(e)}")
            return f"Error importing history: {str(e)}"
    
    def get_gallery_images(self):
        """
        Scans current directory and toolout folders for generated plots.
        """
        # Search for standard plot formats in likely locations
        image_paths = []
        
        # 1. Check Root (where lacathode_roc.png is saved)
        image_paths.extend(glob.glob("*.png"))
        image_paths.extend(glob.glob("*.jpg"))
        
        # 2. Check Tool Output directories recursively
        image_paths.extend(glob.glob("toolout/**/*.png", recursive=True))
        image_paths.extend(glob.glob("toolout/**/*.jpg", recursive=True))
        
        # Sort by modification time (newest first)
        image_paths.sort(key=os.path.getmtime, reverse=True)
        return image_paths
    
    def _background_worker(self, user_input, message_id):
        """Runs the agent loop in a separate thread."""
        self.is_processing = True
        try:
            with self.chat_lock:
                gen = self.active_agent.respond(user_input, message_id)
                for _ in gen: pass
        except Exception as e:
            self.logger.error(f"Background worker crashed: {e}")
        finally:
            self.is_processing = False
    
    def chat_function(self, user_input):
        """Starts the background task and returns immediately."""
        if not user_input.strip():
             return gr.update(value=""), None
        
        # Save prompt to history
        self._save_prompt_to_history(user_input)

        # Fire and Forget Thread
        message_id = uuid.uuid4().hex
        t = threading.Thread(
            target=self._background_worker, 
            args=(user_input, message_id), 
            daemon=True
        )
        t.start()

        # Return immediately (clears box, does NOT update chatbot directly)
        return gr.update(value="", interactive=True)

    def get_agent_history(self, agent_name):
        agent = self.agents.get(agent_name)
        if agent:
            return agent.messages_to_gradio_history()
        return []

    def _save_prompt_to_history(self, text):
        """Saves unique prompts to a JSON file, moving duplicates to the top."""
        history_file = "exports/prompts_history.json"
        os.makedirs("exports", exist_ok=True)
        
        history = []
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                history = json.load(f)
        
        # Remove if exists to handle deduplication (will re-add at start)
        history = [p for p in history if p != text]
        history.insert(0, text) # Most recent at the top
        
        # Keep only last 50 prompts to keep the dropdown clean
        history = history[:50]
        
        with open(history_file, "w") as f:
            json.dump(history, f)
        
        return history

    def _get_prompt_history(self):
        """Loads prompt history for the dropdown."""
        history_file = "exports/prompts_history.json"
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                return json.load(f)
        return []
    
    def run_interactive(self, port=7860):
        with gr.Blocks(fill_height=True) as demo:
            with gr.Tabs():
                    with gr.Tab("Command Center", scale=1):
                        with gr.Row(scale=1):
                            with gr.Column(scale=1):
                                gr.Markdown("### 🧠 Orchestrator (Scientist)")
                                chatbot_active = gr.Chatbot(
                                    value=self.get_agent_history(self.active_agent.__class__.__name__), 
                                    type="messages", 
                                    label="Orchestrator",
                                    scale=1,
                                    autoscroll=False
                                )
                            
                            with gr.Column(scale=1):
                                gr.Markdown("### 🛠️ Analytics (Technician)")
                                chatbot_side = gr.Chatbot(
                                    value=self.get_agent_history("AnalyticsAgent"), 
                                    type="messages",
                                    label="Analytics",
                                    scale=1,
                                    autoscroll=False
                                )

                        with gr.Row(scale=0):
                            with gr.Column(scale=4):
                                msg = gr.Textbox(label="Your Input", placeholder="Type here...", lines=6)
                            
                            with gr.Column(scale=1):
                                submit_btn = gr.Button("Submit", variant="primary")

                                export_btn = gr.DownloadButton("💾 Download Full Session History")

                                prompt_history_dropdown = gr.Dropdown(
                                    choices=self._get_prompt_history(),
                                    label="Latest Prompts",
                                    interactive=True
                                )

                            # Hook up the export button
                            export_btn.click(
                                fn=self.export_all_histories,
                                inputs=None,
                                outputs=export_btn
                            )

                            # When choosing from dropdown, update the textbox
                            prompt_history_dropdown.change(
                                fn=lambda x: x,
                                inputs=[prompt_history_dropdown],
                                outputs=[msg]
                            )

                            # When submitting, refresh the dropdown list
                            # Update your existing msg.submit and submit_btn.click outputs

                            def refresh_dropdown():
                                return gr.update(choices=self._get_prompt_history())
                            
                            submit_event = msg.submit(
                                fn=self.chat_function, 
                                inputs=[msg], 
                                outputs=[msg] 
                            )

                            submit_event.then(
                                fn=refresh_dropdown,
                                inputs=None,
                                outputs=[prompt_history_dropdown]
                            )

                            click_event = submit_btn.click(
                                fn=self.chat_function, 
                                inputs=[msg], 
                                outputs=[msg] 
                            )

                            click_event.then(
                                fn=refresh_dropdown,
                                inputs=None,
                                outputs=[prompt_history_dropdown]
                            )

                            # Timer for background updates (Resume logic)
                            # We can be quite aggressively polling since if there are no changes,
                            # check_background_updates will skip updating the component (no flicker)
                            active_agent_polling_timer = gr.Timer(value=0.3, active=True)
                            active_agent_polling_timer.tick(
                                fn=lambda: self.check_background_updates(self.active_agent.__class__.__name__), 
                                inputs=None,
                                outputs=[chatbot_active, msg]
                            )

                            side_agent_polling_timer = gr.Timer(value=5.0, active=True)
                            side_agent_polling_timer.tick(
                                fn=lambda: self.check_background_updates("AnalyticsAgent"),
                                inputs=None,
                                outputs=[chatbot_side, msg]
                            )

                    with gr.Tab("Visualizations", scale=1):
                        gr.Markdown("### Generated Plots & Figures")
                        refresh_btn = gr.Button("🔄 Refresh Gallery")
                        gallery = gr.Gallery(
                            label="Generated Images", 
                            show_label=False, 
                            elem_id="gallery", 
                            columns=[3], 
                            rows=[2], 
                            object_fit="contain", 
                            height="auto"
                        )
                        
                        # Load images on click
                        refresh_btn.click(
                            fn=self.get_gallery_images,
                            inputs=None,
                            outputs=gallery
                        )
                    
                    with gr.Tab("Tools", scale=1):
                        # Import Session
                        gr.Markdown("### 📂 Import Previous Session")
                        def handle_import(file_obj):
                            # Run the import
                            status = self.import_all_histories(file_obj)
                            
                            # Return updates for both chatbots to show restored history immediately
                            return (
                                gr.update(placeholder=f"System: {status}"), # Feedback in the text box
                                self.get_agent_history(self.active_agent.__class__.__name__), # Refresh Active
                                self.get_agent_history("AnalyticsAgent") # Refresh Side
                            )

                        import_btn = gr.File(
                            label="📂 Restore Session", 
                            file_types=[".json"], 
                            file_count="single",
                            type="filepath" # Passes the file path string to the function
                        )
                                    
                        import_btn.upload(
                            fn=handle_import,
                            inputs=[import_btn],
                            outputs=[msg, chatbot_active, chatbot_side]
                        )
                    
                        gr.Markdown("### 📚 External Knowledge Logs (Gemma)")
                        gr.Markdown("This log shows the specific questions asked to the external knowledge base and the responses received.")
                        
                        refresh_tools_btn = gr.Button("🔄 Refresh Logs")
                        tools_log_display = gr.Markdown("No logs yet...")
                        
                        # Load initial state
                        demo.load(
                            fn=get_runtime_history_file,
                            inputs=None,
                            outputs=tools_log_display
                        )

                        # Manual Refresh
                        refresh_tools_btn.click(
                            fn=get_runtime_history_file,
                            inputs=None,
                            outputs=tools_log_display
                        )
            demo.load(
                fn=lambda: self.check_background_updates(self.active_agent.__class__.__name__),
                inputs=None,
                outputs=[chatbot_active, msg]
            )

            demo.load(
                fn=lambda: self.check_background_updates("AnalyticsAgent"),
                inputs=None,
                outputs=[chatbot_side, msg]
            )

        demo.launch(share=False, server_port=port)