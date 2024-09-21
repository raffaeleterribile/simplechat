import torch
from typing import Final
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Phi3Config, Phi3Model

MODEL_ID : Final[str] = "microsoft/Phi-3-mini-128k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
	"What a wonderful life"
]

tokens = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(tokens)

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, torch_dtype="auto")

outputs = model(**inputs)
# print(outputs.last_hidden_state.shape)
print(outputs.logits.shape)
print(outputs.logits)
print(outputs)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

print(model.config.id2label)

config = Phi3Config(MODEL_ID)

# model = Phi3Model(config)

model = Phi3Model.from_pretrained(MODEL_ID)

# model.save_pretrained("Phi-3-mini-128k-instruct")

sequences = ["Hello!", "Cool.", "Nice!"]

tokenized_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
print(tokenized_sequences)

encoded_sequences = tokenized_sequences.input_ids
print(encoded_sequences)

model_inputs = torch.tensor(encoded_sequences)
print(model_inputs)

output = model(model_inputs)
print(output)

# tokenizer.save_pretrained("Phi-3-mini-128k-instruct")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
print(tokens)

vocabulary_indexes = tokenizer.convert_tokens_to_ids(tokens)
print(vocabulary_indexes)

decoded_string = tokenizer.decode(vocabulary_indexes)
print(decoded_string)

# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Will truncate the sequences that are longer than the model max length
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
# model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))

