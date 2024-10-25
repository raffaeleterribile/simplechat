""" Demonstrates a simple chat interface using gRPC. """

import gradio as gr
import grpc
import simple_chat_pb2
import simple_chat_pb2_grpc

def generate_sync_message(input):
	""" Generates a response to a given input text in sync mode. """
	# Connessione al server gRPC
	with grpc.insecure_channel('localhost:50051') as channel:
		stub = simple_chat_pb2_grpc.ChatServiceStub(channel)

		# Creazione del messaggio da inviare
		request = simple_chat_pb2.ChatMessage(message=input)

		# Invio del messaggio al server e ricezione della risposta
		response = stub.send_message(request)

		print(f"Sync Server response: {response.message}")

		return response.message

def generate_stream_message(input):
	""" Generates a response to a given input text in streaming mode. """
	with grpc.insecure_channel('localhost:50051') as channel:
		stub = simple_chat_pb2_grpc.ChatServiceStub(channel)

		# Creazione del messaggio da inviare
		request = simple_chat_pb2.ChatMessage(message=input)

		# Invio del messaggio al server e ricezione della risposta in modalità streaming
		response_iterator = stub.send_stream_message(request)
		for partial_response in response_iterator:
			response = partial_response
			print(f"Streaming Server response: {response.message}")
			yield response.message

		print(f"Complete Server response: {response.message}")

		return response.message

def generate_message(input, choice):
	""" Generates a response to a given input text. """
	if choice == "Sync":
		return generate_sync_message(input)
	else:
		text_iterator = generate_stream_message(input)
		for text in text_iterator:
			yield text

choice = gr.Radio(["Sync", "Stream"], value="Stream", label="Choose the type of response")
input_text = gr.Textbox(lines=1, label="Input Text")
output_text = gr.Textbox(lines=1, label="Output Text")

demo = gr.Interface(fn=generate_message, inputs=[input_text, choice], outputs=output_text,
					title="Chat Interface", description="Generates a response to a given input text.")

demo.launch()
