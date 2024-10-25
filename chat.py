""" Demonstrates a simple chat interface. """

import gradio as gr
from ai import Generator

generator = Generator()

demo = gr.ChatInterface(generator.generate, type="messages")
demo.launch()
