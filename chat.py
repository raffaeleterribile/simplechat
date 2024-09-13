""" Demonstrates a simple chat interface using the RedPajama-INCITE-Chat-3B-v1 model. """

import gradio as gr
from ai import Generator

generator = Generator()

demo = gr.ChatInterface(generator.generate)
demo.launch()
