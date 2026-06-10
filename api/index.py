from fastapi import FastAPI
import gradio as gr
from app import build_app

app = FastAPI()

demo = build_app()
gr.mount_gradio_app(app, demo, path="/")
