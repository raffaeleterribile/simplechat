""" Classes used to generate responses """
from enum import Enum
from threading import Thread
from typing import Final
import re
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

# The default was "togethercomputer/RedPajama-INCITE-Chat-3B-v1" and his default stop tokens are [29, 0]
MODEL_ID : Final[str] = "microsoft/Phi-3-mini-128k-instruct"
STREAMER_TIMEOUT : Final[float] = 100.0 # seconds. The default was 10. but, with 10., the app crashes
# MAX_OUTPUT_LENGTH : Final[int] = 4096 # The default was 1024

""" SPECIAL TOKENS
SYSTEM_TOKEN = "<|system|>"
ASSISTANT_TOKEN = "<|assistant|>"
USER_TOKEN = "<|user|>"
UNKNOWN_TOKEN = "<unk>"
BEGIN_OF_SENTENCE_TOKEN = "<s>"
END_OF_SENTENCE_TOKEN = "</s>"
END_TOKEN = "<|end|>"
PAD_TOKEN = "<|endoftext|>"
END_OF_TEXT_TOKEN = "<|endoftext|>"
PLACEHOLDER_1_TOKEN = "<|placeholder1|>"
PLACEHOLDER_2_TOKEN = "<|placeholder2|>"
PLACEHOLDER_3_TOKEN = "<|placeholder3|>"
PLACEHOLDER_4_TOKEN = "<|placeholder4|>"
PLACEHOLDER_5_TOKEN = "<|placeholder5|>"
PLACEHOLDER_6_TOKEN = "<|placeholder6|>"

SPECIAL_TOKENS = {
	"SYSTEM_TOKEN": 32006,
	"ASSISTANT_TOKEN": 32001,
	"USER_TOKEN": 32010,
	"UNKNOWN_TOKEN": 0,
	"BEGIN_OF_SENTENCE_TOKEN": 1,
	"END_OF_SENTENCE_TOKEN": 2,
	"END_TOKEN": 32007,
	"PAD_TOKEN": 32000,
	"END_OF_TEXT_TOKEN": 32000,
	"PLACEHOLDER_1_TOKEN": 32002,
	"PLACEHOLDER_2_TOKEN": 32003,
	"PLACEHOLDER_3_TOKEN": 32004,
	"PLACEHOLDER_4_TOKEN": 32005,
	"PLACEHOLDER_5_TOKEN": 32008,
	"PLACEHOLDER_6_TOKEN": 32009
} """

class SpecialTokens(Enum):
	""" Special tokens used by the model. """
	SYSTEM_TOKEN = "<|system|>"
	ASSISTANT_TOKEN = "<|assistant|>"
	USER_TOKEN = "<|user|>"
	UNKNOWN_TOKEN = "<unk>"
	BEGIN_OF_SENTENCE_TOKEN = "<s>"
	END_OF_SENTENCE_TOKEN = "</s>"
	END_TOKEN = "<|end|>"
	PAD_TOKEN = "<|endoftext|>"
	END_OF_TEXT_TOKEN = "<|endoftext|>"
	PLACEHOLDER_1_TOKEN = "<|placeholder1|>"
	PLACEHOLDER_2_TOKEN = "<|placeholder2|>"
	PLACEHOLDER_3_TOKEN = "<|placeholder3|>"
	PLACEHOLDER_4_TOKEN = "<|placeholder4|>"
	PLACEHOLDER_5_TOKEN = "<|placeholder5|>"
	PLACEHOLDER_6_TOKEN = "<|placeholder6|>"

	def __call__(self) -> torch.LongTensor:
		special_tokens_ids = {
			SpecialTokens.SYSTEM_TOKEN: torch.LongTensor(32006),
			SpecialTokens.ASSISTANT_TOKEN: torch.LongTensor(32001),
			SpecialTokens.USER_TOKEN: torch.LongTensor(32010),
			SpecialTokens.UNKNOWN_TOKEN: torch.LongTensor(0),
			SpecialTokens.BEGIN_OF_SENTENCE_TOKEN: torch.LongTensor(1),
			SpecialTokens.END_OF_SENTENCE_TOKEN: torch.LongTensor(2),
			SpecialTokens.END_TOKEN: torch.LongTensor(32007),
			SpecialTokens.PAD_TOKEN: torch.LongTensor(32000),
			SpecialTokens.END_OF_TEXT_TOKEN: torch.LongTensor(32000),
			SpecialTokens.PLACEHOLDER_1_TOKEN: torch.LongTensor(32002),
			SpecialTokens.PLACEHOLDER_2_TOKEN: torch.LongTensor(32003),
			SpecialTokens.PLACEHOLDER_3_TOKEN: torch.LongTensor(32004),
			SpecialTokens.PLACEHOLDER_4_TOKEN: torch.LongTensor(32005),
			SpecialTokens.PLACEHOLDER_5_TOKEN: torch.LongTensor(32008),
			SpecialTokens.PLACEHOLDER_6_TOKEN: torch.LongTensor(32009)
		}

		return special_tokens_ids[self]

class StopOnTokens(StoppingCriteria):
	""" Stop when a stop token is generated. """

	def __init__(self, stop_ids: torch.LongTensor) -> None:
		super().__init__()
		self.stop_ids = stop_ids

	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
		for stop_id in self.stop_ids:
			if input_ids[0][-1] == stop_id:
				return True
		return False

class Generator:
	""" Wrapper for the AI model. """

	def __init__(self):
		super().__init__()
		self.history = []

		# Ottieni il riferimento al dispositivo
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		device_map = "cuda" if torch.cuda.is_available() else "cpu"

		# Stampa il dispositivo per conferma
		print(f"Using device: {self.device}")

		self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True,
												 use_chat_template=True, add_generation_prompt=True)

		# With torch_dtype="auto", the model will load much faster
		# Notes: If you want to use flash attention, call AutoModelForCausalLM.from_pretrained() with attn_implementation="flash_attention_2"
		self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
													device_map=device_map,
													torch_dtype="auto",
													trust_remote_code=True)
		# self.model = self.model.to('cuda:0')
		self.model = self.model.to(self.device)

		self.pipeline = pipeline("text-generation",
			tokenizer=self.tokenizer, model=self.model,
			torch_dtype="auto", trust_remote_code=True)

	def generate(self, message, history):
		""" Generate a response to a message. """
		message = self.clean_text(message)

		messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

		for history_message in history:
			history_message['content'] = self.clean_text(history_message['content'])

		messages.extend(history)
		messages.append({"role": "user", "content": message})

		stop = StopOnTokens(SpecialTokens.END_TOKEN()) # 32000 - END_OF_TEXT_TOKEN

		messages = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

		model_inputs = self.tokenizer([messages], return_tensors="pt").to(self.device)

		streamer = TextIteratorStreamer(self.tokenizer, timeout=STREAMER_TIMEOUT, skip_prompt=True, skip_special_tokens=False)
		generate_kwargs = dict(
				model_inputs,
				streamer=streamer,
				max_new_tokens=1024,
				do_sample=True,
				top_p=0.95,
				top_k=1000,
				temperature=1.0,
				num_beams=1,
				stopping_criteria=StoppingCriteriaList([stop])
			)
		thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
		thread.start()

		partial_message = ""
		for generated_token in streamer:
			new_token = self.clean_text(generated_token)

			partial_message += new_token
			print(f"generated_token:\n\t\"{generated_token}\"\nnew_token:\n\t\"{new_token}\"\npartial_message:\n\t\"{partial_message}\"")
			yield partial_message

	def clean_text(self, original_text):
		""" Clean the text by removing special tokens and extra spaces. """
		if original_text is None:
			return ""

		# Rimuove i token speciali
		cleaned_text = original_text
		for special_token in SpecialTokens:
			cleaned_text = cleaned_text.replace(special_token.value, "")

		# Rimuove i caratteri di controllo
		cleaned_text = re.sub(r'[\x00-\x1F\x7F]', '', cleaned_text)

		# Sostituisce i caratteri di tipo whitespace con uno spazio
		cleaned_text = re.sub(r'\s', ' ', cleaned_text)

		while "  " in cleaned_text:
			cleaned_text = cleaned_text.replace("  ", " ")

		return cleaned_text
