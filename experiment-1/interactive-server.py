from argparse import ArgumentParser
import glob
import os
import random
from threading import Thread
import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig, TextIteratorStreamer
from numeric_fusion_adapter import NumericFusionAdapter

# Mac support for MPS
device = "mps" if torch.backends.mps.is_available() else "cpu"
# GPU support
if torch.cuda.is_available():
    device = "cuda"

base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
lora_checkpoint = "lora_lhco/checkpoint-*"  # path to LoRA weights

arg_parser = ArgumentParser()
arg_parser.add_argument("--use_checkpoint", type=int, default=0, required=False, help="Give custom checkpoint number to use, or 0 for latest.")
arg_parser.add_argument("--use_numeric", action='store_true', help="Use numeric feature token.", default=True)
args = arg_parser.parse_args()

# Get the latest folder in the checkpoint directory
list_of_dirs = glob.glob(lora_checkpoint)
lora_checkpoint = max(list_of_dirs, key=os.path.getctime)

if args.use_checkpoint != 0:
    lora_checkpoint = f"lora_lhco/checkpoint-{args.use_checkpoint}"

print(f"Using LoRA checkpoint: {lora_checkpoint}")

"""
Special token for numeric features
"""
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token  # ensure padding

if args.use_numeric:
    JET_FEATURE_TOKEN_STR = "<JET_FEATURES>"
    tokenizer.add_special_tokens({"additional_special_tokens": [JET_FEATURE_TOKEN_STR]})
    JET_FEATURE_TOKEN_ID = tokenizer.convert_tokens_to_ids(JET_FEATURE_TOKEN_STR)

    print(f"Numeric feature token: {JET_FEATURE_TOKEN_STR} with ID {JET_FEATURE_TOKEN_ID}.\nMake sure this is consistent with training !")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # replaces torch_dtype
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    device_map="auto",
    quantization_config=bnb_config,  # replaces load_in_4bit argument
)
# Resize token embeddings in case new tokens were added to tokenizer
model.resize_token_embeddings(len(tokenizer))

# Initialize and Load Numeric Fusion Adapter
numeric_dim = 16 
hidden_size = model.config.hidden_size
model.numeric_fusion_adapter = NumericFusionAdapter(hidden_size, numeric_dim, dtype=torch.float16, device=device).to(device)

# Resize token embeddings in case new tokens were added to tokenizer
model.resize_token_embeddings(len(tokenizer))

# Load the trained adapter weights
adapter_weights_path = os.path.join(lora_checkpoint, "numeric_fusion_adapter.bin")
if args.use_numeric:
    try:
        adapter_state_dict = torch.load(adapter_weights_path, map_location=device)
        model.numeric_fusion_adapter.load_state_dict(adapter_state_dict)
        print(f"Loaded NumericFusionAdapter weights from {adapter_weights_path}")
    except FileNotFoundError:
        print(f"WARNING: NumericFusionAdapter weights not found at {adapter_weights_path}. Adapter is using random weights!")

# Apply LoRA weights
model = PeftModel.from_pretrained(
    model,
    lora_checkpoint,
    device_map="auto",
    adapter_name=lora_checkpoint
)

# PEFT doesn't automatically set the model as "active" with PeftModel.from_pretrained
# Adapters still work, but for disabling/enabling we need to set it active first.
print(model.peft_config)
model.set_adapter(lora_checkpoint)

# print type of model to verify
print(f"Model type after applying LoRA: {type(model)}")

# Optionally merge the LoRA
# Merge LoRA weights into base model
# model = model.merge_and_unload()

model.to(device)
model.eval()

print("Model and tokenizer loaded.")

"""
Text Generation Function
"""
def stream_chat(message: str, history: list):
    """
    This function handles the chat interaction, applying the
    chat template and using a streamer for token-by-token output.
    """
    
    messages = history.copy() if history else []

    messages = [{"role": "system", "content": 
                 "You are a Physics jet expert. You are fine tuned on a dataset of jet physics interactionns. "
                 "First message you receive triggers a LoRA to make you give classification such as 'signal' or 'background'. "
                 "This LoRA is disabled in all subsequent messages. But try to make explanations based on the classification. "
                 }
                 ] + messages
   
    messages.append({"role": "user", "content": message})

    # Message index 1 is always tool call
    random_tool_call_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=9))

    messages = messages[:1] + [{
                "role": "tool_results", 
                "tool_call_id": random_tool_call_id, 
                "content": '{"result": "' + JET_FEATURE_TOKEN_STR + '"}'
            }] + messages[1:]

    print("Messages so far:", messages)

    # 3. Apply chat template to format messages
    def jet_feature_tool():
        """
        Tool that enables LoRA for the next message.

        Args:
            None
        """
        print("Activating LoRA for next message.")


    prompt = tokenizer.apply_chat_template(
        messages,
        tools=[jet_feature_tool],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Print the final prompt for debugging
    print("Final prompt to model:")
    print(prompt)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=False,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Warn if input is too long
    if input_ids.shape[1] > 4096:
        print(f"WARNING: Input length {input_ids.shape[1]} exceeds model's max length of 4096 tokens.")

    # 4. Set up the streamer
    #    `TextIteratorStreamer` is used for streaming in a
    #    non-blocking way, perfect for Gradio.
    streamer = TextIteratorStreamer(
        tokenizer, 
        timeout=180.0, 
        skip_prompt=True, 
        skip_special_tokens=True
    )

    # 5. Define generation arguments
    generate_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=4096,
        #temperature=0.8,
        #top_p=0.8,
        #do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    def generate(**kwargs):
        print(messages)
        if len(messages) < 4: # First message + tool call + system = 3
            resp = model.generate(**kwargs)
        else:
            with model.disable_adapter():
                resp = model.generate(**kwargs)

        # Check if there is a tool call
        decoded = tokenizer.decode(resp[0], skip_special_tokens=False)

        if '[TOOL_CALL]' in decoded:
            print("Tool call detected in generation output.")
            # Here you would parse the tool call and execute the tool
            # For simplicity, we just print a message
            print("Executing tool... (not implemented in this example)")
        
        # Run the generation again with tool call result

        return resp

    # 6. Start generation in a separate thread
    #    This is crucial for streaming; it allows the `yield`
    #    loop to run while the model generates tokens.
    thread = Thread(target=generate, kwargs=generate_kwargs)
    thread.start()

    # 7. Yield tokens as they become available
    partial_response = ""
    for new_text in streamer:
        if new_text: # Ensure not yielding empty strings
            partial_response += new_text
            yield partial_response


# --- 3. Create and Launch the Gradio Chat Interface ---

# `gr.ChatInterface` handles the UI components (Chatbot, Textbox, Button)
# and the state (history) for you.
demo = gr.ChatInterface(
    fn=stream_chat,
    title=f"{base_model_name} + LoRA Chat",
    description="Chat with Mistral-7B-Instruct-v0.3 model fine-tuned with LoRA on LHCO dataset.",
    theme="soft",
    type="messages"
)

print("Launching Gradio interface... (Check your console for the URL)")
demo.launch(share=False) # share=True creates a public link