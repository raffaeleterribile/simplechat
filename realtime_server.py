""" Server realtime using gRPC """
import grpc
from concurrent import futures
import time
import simple_chat_pb2
import simple_chat_pb2_grpc
from ai import Generator

class ChatService(simple_chat_pb2_grpc.ChatServiceServicer):
	def __init__(self):
		super().__init__()
		self.generator = Generator()
		self.history = []

	def send_message(self, request, context):
		""" Genera una risposta ad un messaggio """
		text_iterator = self.generator.generate(request.message, self.history)

		text = ""
		for partial_text in text_iterator:
			text = partial_text

		self.history.append({"role": "user", "content": request.message})
		self.history.append({"role": "assistant", "content": text})

		response = simple_chat_pb2.ChatMessage()
		response.message = text
		return response

	def send_stream_message(self, request, context):
		""" Genera una risposta ad un messaggio in modalità streaming """
		text_iterator = self.generator.generate(request.message, self.history)

		try:
			text = ""
			for partial_text in text_iterator:
				text = partial_text
				response = simple_chat_pb2.ChatMessage()
				response.message = text
				yield response
		except StopIteration:
			self.history.append({"role": "user", "content": request.message})
			self.history.append({"role": "assistant", "content": text})

def serve():
	""" Avvia il server gRPC """
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
	simple_chat_pb2_grpc.add_ChatServiceServicer_to_server(ChatService(), server)
	server.add_insecure_port('[::]:50051')
	server.start()
	try:
		while True:
			time.sleep(86400)
	except KeyboardInterrupt:
		server.stop(0)

if __name__ == '__main__':
	serve()
