""" Classes used to generate responses """

from threading import Thread
from typing import Final
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

# The default was "togethercomputer/RedPajama-INCITE-Chat-3B-v1" and his default stop tokens are [29, 0]
MODEL_ID : Final[str] = "microsoft/Phi-3-mini-128k-instruct"
STREAMER_TIMEOUT : Final[float] = 100.0 # seconds. The default was 10. but, with 10., the app crashes
# MAX_OUTPUT_LENGTH : Final[int] = 4096 # The default was 1024

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
	SYSTEM_TOKEN: 32006,
	ASSISTANT_TOKEN: 32001,
	USER_TOKEN: 32010,
	UNKNOWN_TOKEN: 0,
	BEGIN_OF_SENTENCE_TOKEN: 1,
	END_OF_SENTENCE_TOKEN: 2,
	END_TOKEN: 32007,
	PAD_TOKEN: 32000,
	END_OF_TEXT_TOKEN: 32000,
	PLACEHOLDER_1_TOKEN: 32002,
	PLACEHOLDER_2_TOKEN: 32003,
	PLACEHOLDER_3_TOKEN: 32004,
	PLACEHOLDER_4_TOKEN: 32005,
	PLACEHOLDER_5_TOKEN: 32008,
	PLACEHOLDER_6_TOKEN: 32009
}

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

		self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

		# With torch_dtype="auto", the model will load much faster
		self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype="auto")
		# self.model = self.model.to('cuda:0')

	def generate(self, message, history):
		""" Generate a response to a message. """
		history_transformer_format = history + [[message, ""]]
		stop = StopOnTokens([32000])

     	# "chat_template": "{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}"

		# for message in messages:
		# 	if message['role'] == 'system':
		# 		history_transformer_format.append('<|system|>\n' + message['content'] + '<|end|>\n', ""])
		# 	elif message['role'] == 'user':
		# 		history_transformer_format.append(["", '<|user|>\n' + message['content'] + '<|end|>\n'])
		# 	elif message['role'] == 'assistant':
		# 		history_transformer_format[-1][1] = '<|assistant|>\n' + message['content'] + '<|end|>\n'

		# Original code for the "togethercomputer/RedPajama-INCITE-Chat-3B-v1" model
		messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])
				for item in history_transformer_format])

		# My code for the "microsoft/Phi-3-mini-128k-instruct" model
		# messages = "".join(["".join(["<|user|>\n " + item[0] + "<|end|>\n", "<|assistant|>\n " + item[1] + "<|end|>\n"])
		# 		for item in history_transformer_format])

		model_inputs = self.tokenizer([messages], return_tensors="pt") #.to("cuda")
		streamer = TextIteratorStreamer(self.tokenizer, timeout=STREAMER_TIMEOUT, skip_prompt=True, skip_special_tokens=True)
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
		for new_token in streamer:
			if new_token != '<':
				partial_message += new_token
				yield partial_message
