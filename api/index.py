from fastapi import FastAPI
import gradio as gr
from app import build_app, _build_theme, CSS, FORCE_LIGHT_JS

app = FastAPI()

demo = build_app()
# Gradio 6: theme/css must be passed at mount/launch time, not the Blocks constructor.
gr.mount_gradio_app(app, demo, path="/", theme=_build_theme(), css=CSS, js=FORCE_LIGHT_JS)
