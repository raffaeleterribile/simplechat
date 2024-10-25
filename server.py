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
def chat(message: Message) -> Message:
	""" Genera una risposta ad un messaggio """
	response_iterator = generator.generate(message.message, history)

	response = ""
	for partial_response in response_iterator:
		response = partial_response

	history.append({"role": "user", "content": message.message})
	history.append({"role": "assistant", "content": response})
	return Message(message=response)

if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=8000)
