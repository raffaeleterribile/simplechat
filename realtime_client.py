""" Client realtime using gRPC """
import grpc
import simple_chat_pb2
import simple_chat_pb2_grpc
import sys

def run():
	""" Invia un messaggio al server gRPC """
	# Verifica se è stato passato un parametro di input
	if len(sys.argv) < 3:
		print("Usage: python realtime_client.py <sync|stream> <message>")
		return

	# Recupera il tipo di interazione da eseguire: sincrona ("Sync") o streaming ("Stream")
	chat_type = sys.argv[1]
	if (chat_type != "sync") and (chat_type != "stream"):
		print("Usage: python realtime_client.py <sync|stream> <message>")
		return

	# Recupera il messaggio dall'input dell'utente
	user_message = sys.argv[2]

	if chat_type == "sync":
		generate_sync_message(user_message)
	else:
		generate_stream_message(user_message)

def generate_sync_message(user_message):
	""" Genera messaggi sincroni """

	# Connessione al server gRPC
	with grpc.insecure_channel('localhost:50051') as channel:
		stub = simple_chat_pb2_grpc.ChatServiceStub(channel)

		# Creazione del messaggio da inviare
		request = simple_chat_pb2.ChatMessage(message=user_message)

		# Invio del messaggio al server e ricezione della risposta
		response = stub.send_message(request)

		print(f"Sync Server response: {response.message}")

def generate_stream_message(user_message):
	""" Genera messaggi in strwaming """

	# Connessione al server gRPC
	with grpc.insecure_channel('localhost:50051') as channel:
		stub = simple_chat_pb2_grpc.ChatServiceStub(channel)

		# Creazione del messaggio da inviare
		request = simple_chat_pb2.ChatMessage(message=user_message)

		# Invio del messaggio al server e ricezione della risposta in modalità streaming
		response_iterator = stub.send_stream_message(request)
		for partial_response in response_iterator:
			response = partial_response
			print(f"Streaming Server response: {response.message}")

		print(f"Complete Server response: {response.message}")

if __name__ == '__main__':
	run()
