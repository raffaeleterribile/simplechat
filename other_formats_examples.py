from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

GGUF_MODEL_ID = "Phi-3-mini-4k-instruct-gguf" # from https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/tree/main

# Load the tokenizer
# gguf_tokenizer = AutoTokenizer.from_pretrained(GGUF_MODEL_ID)

# Load the model
# gguf_model = AutoModelForCausalLM.from_pretrained(GGUF_MODEL_ID, format="gguf")

""" Example code for the GGUF format
from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
	filename="Phi-3-mini-4k-instruct-fp16.gguf",
)

llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	]
)"""

ONNX_MODEL_ID = "Phi-3-mini-4k-instruct-onnx" # from https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4

# Load the tokenizer
# onnx_tokenizer = AutoTokenizer.from_pretrained(ONNX_MODEL_ID)

# Load the model
# onnx_model = AutoModelForCausalLM.from_pretrained(ONNX_MODEL_ID, format="onnx")

""" Example code for the ONNX format
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct-onnx", trust_remote_code=True)

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct-onnx", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct-onnx", trust_remote_code=True)

"""