""" Examples with Phi3 Instructions model. """
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, pipeline
from safetensors.torch import load_file

MODEL_ID = "microsoft/Phi-3-vision-128k-instruct"

user_folder = os.path.expanduser("~")
MODEL_FOLDER = user_folder + "/.cache/huggingface/hub/models--microsoft--Phi-3-mini-128k-instruct/snapshots/a90b62ae09941edff87a90ced39ba5807e6b2ade"

torch.random.manual_seed(0) # Set seed for reproducibility

# Use a pipeline as a high-level helper
messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model=MODEL_ID, torch_dtype="auto", trust_remote_code=True, _attn_implementation="default")
result = pipe(messages)
print("Phi with a pipeline: ", result)
pipe = None

# Load model with model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained( 
    MODEL_ID,
    # device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True
)

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}
]

pipe = pipeline( 
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype="auto"
)

generation_args = { 
    "max_new_tokens": 500, 
    "return_full_text": False, 
    "temperature": 0.0, 
    "do_sample": False, 
}

output = pipe(messages, **generation_args) 
print("Phi with model and tokenizer: ", output[0]['generated_text'])
pipe = None

# Specifica il percorso del file safetensors
# fc534bed68dc12e48d1f28d144bfb2239bd1ef4f5d5892d68e9d1f85731b8c12 Parte 1 di 2
# d08786c6cca821d229fd6049c3a691a51bcfa0d29b78289dbc607badd92b2bdc Parte 2 di 2
# file_path = user_folder + "/.cache/huggingface/hub/models--microsoft--Phi-3-mini-128k-instruct/blobs/fc534bed68dc12e48d1f28d144bfb2239bd1ef4f5d5892d68e9d1f85731b8c12"

# Carica i tensori dal file
# tensors = load_file(file_path)
# tensors = None

# Ora puoi utilizzare i tensori con PyTorch
# for name, tensor in tensors.items():
#     print(f"{name}: {tensor.shape}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_FOLDER,
    # device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    # _attn_implementation="flash_attention_2"
)
model_inputs = tokenizer("How many planets there are in the solar system?", return_tensors="pt")
input_ids = model_inputs.input_ids
result = model.generate(input_ids)
decoded = tokenizer.batch_decode(result)
output = decoded[0]
print("Phi loaded directly from a folder: ", output)
model = None

processor = AutoProcessor.from_pretrained(MODEL_FOLDER, torch_dtype="auto", trust_remote_code=True)
output = processor("This is a test")
print("Phi loaded directly from a folder with a processor: ", output)
processor = None