from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TextIteratorStreamer
import torch
import gradio as gr
import json
import uuid
import gc

from framework.rag_engine import RAGEngine
from framework.orchestrator_agent import OrchestratorAgent
from framework.utilities.cuda_ram_debug import log_cuda_memory

max_memory = {
    0: "7.2GB",
    "cpu": "20GB"
}

class Framework:
    def __init__(self, *args, **kwargs):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True, # Keeps VRAM low
            bnb_4bit_quant_type="nf4",      # Best for pre-trained weights
            bnb_4bit_compute_dtype=compute_dtype 
        )
        
        self.base_model_name = kwargs.get('base_model_name', 'Qwen/Qwen3-4B')
        
        self.model = None
        self.load_model() # This sets self.model

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
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
            rag_engine=self.rag_engine,
            model_loader=self.load_model,
            model_unloader=self.unload_model
        )

    def load_model(self):
        print("Framework: Loading model reference...")
        log_cuda_memory("CUDA Memory Before loading model")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            max_memory=max_memory,
            quantization_config=self.bnb_config,  # replaces load_in_4bit argument
        )
        log_cuda_memory("CUDA Memory After loading model")

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
            gc.collect()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            
            torch.cuda.empty_cache()

            log_cuda_memory("CUDA Memory After unloading model for heavy tool")
    
    def chat_function(self, user_input, chat_history):
        message_id = uuid.uuid4().hex
        parsed_generator = self.orchestrator_agent.respond(user_input, message_id, chat_history)

        for multiple_parsed in parsed_generator:
            response = []
            for parsed in multiple_parsed:
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

                if parsed.get("tool_result"):
                    response.append(gr.ChatMessage(
                        role="assistant", 
                        content=f"Tool Result:\n{parsed['tool_result']}",
                        metadata={"title": "Tool Result", "id": message_id}
                    ))
                
                if parsed["content"]:
                    response.append(gr.ChatMessage(
                        role="assistant", 
                        content=parsed["content"],
                        metadata={"id": message_id}
                    ))

            yield response

    def run_interactive(self):
        with gr.Blocks(fill_height=True) as demo:
            chatbot = gr.Chatbot(
                type="messages",
                scale=1
            )

            chatbot.clear(fn=lambda: self.orchestrator_agent.messages.clear())

            gr.ChatInterface(
                fn=self.chat_function,
                title="Interactive Chat with Orchestrator Model",
                description="Chat interface for interacting with the orchestrator model.",
                type="messages",
                chatbot=chatbot,
            )

        demo.launch(share=False)

if __name__ == "__main__":
    framework = Framework(base_model_name='Qwen/Qwen3-4B')
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

    

