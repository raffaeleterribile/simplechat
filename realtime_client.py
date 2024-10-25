""" Client realtime using gRPC """
import grpc
import simple_chat_pb2
import simple_chat_pb2_grpc
import sys

def run():
	# Verifica se è stato passato un parametro di input
	if len(sys.argv) < 2:
		print("Usage: python realtime_client.py <message>")
		return

	# Recupera il messaggio dall'input dell'utente
	user_message = sys.argv[1]

	# Connessione al server gRPC
	with grpc.insecure_channel('localhost:50051') as channel:
		stub = simple_chat_pb2_grpc.ChatServiceStub(channel)

		# Creazione del messaggio da inviare
		request = simple_chat_pb2.ChatMessage(message=user_message)

		# Invio del messaggio al server e ricezione della risposta
		response = stub.send_message(request)

		print(f"Sync Server response: {response.message}")

		# Invio del messaggio al server e ricezione della risposta in modalità streaming
		response_iterator = stub.send_stream_message(request)
		for partial_response in response_iterator:
			response = partial_response
			print(f"Streaming Server response: {response.message}")

		print(f"Complete Server response: {response.message}")

if __name__ == '__main__':
	run()
