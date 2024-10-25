""" Endpoint per la chat """

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ai import Generator

app = FastAPI()
generator = Generator()
history = []

@app.get("/")
def root():
	""" Home page"""
	return "Hello World!"

class Message(BaseModel):
	""" Modello per il messaggio """
	message: str

@app.post("/chat/")
def chat(request: Message) -> Message:
	""" Genera una risposta ad un messaggio """
	text_iterator = generator.generate(request.message, history)

	text = ""
	for partial_text in text_iterator:
		text = partial_text

	history.append({"role": "user", "content": request.message})
	history.append({"role": "assistant", "content": text})

	response = Message(message=text)
	return response

if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=8000)
