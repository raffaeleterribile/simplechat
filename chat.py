""" Demonstrates a simple chat interface using the RedPajama-INCITE-Chat-3B-v1 model. """

from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import gradio as gr

MODEL_ID = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype="auto") # With torch_dtype="auto", the model will be loaded much faster
# tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
# model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.float16)
# model = model.to('cuda:0')

STREAMER_TIMEOUT = 100.0 # seconds. The default was 10. but, with 10., the app crashes

class StopOnTokens(StoppingCriteria):
    """ Stop when a stop token is generated. """

    def __init__(self, stop_ids: torch.LongTensor) -> None:
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # stop_ids = [29, 0]
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def predict(message, history):
    """ Generate a response to a message. """
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens([29, 0])

    messages = "".join(["".join(["\n<human>:"+item[0], "\n<bot>:"+item[1]])
                for item in history_transformer_format])

    model_inputs = tokenizer([messages], return_tensors="pt") #.to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=STREAMER_TIMEOUT, skip_prompt=True, skip_special_tokens=True)
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
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    partial_message = ""
    for new_token in streamer:
        if new_token != '<':
            partial_message += new_token
            yield partial_message

app = gr.ChatInterface(predict)
app.launch()
