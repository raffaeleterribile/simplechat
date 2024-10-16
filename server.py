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
	return "Hello World!"

class Message(BaseModel):
	message: str

@app.post("/chat/")
def chat(message: Message) -> Message:
	response = generator.generate(message.message, history)
	history.append([message, response])
	return Message(message=response)

if __name__ == "__main__":
	uvicorn.run(app, host="127.0.0.1", port=8000)
