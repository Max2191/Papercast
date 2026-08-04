from __future__ import annotations

import traceback

import gradio as gr

from src.pipeline import (
    audio_from_state,
    narration_from_state,
    process_paper,
    summarize_state,
)
from src.sections import section_table

try:
    import spaces

    gpu_task = spaces.GPU
except ImportError:
    def gpu_task(function):
        return function


def _friendly_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


@gpu_task
def handle_process(uploaded_file, source_text, parser_label, page_limit):
    try:
        parser_mode = "granite" if parser_label.startswith("Granite") else "standard"
        paper = process_paper(
            uploaded_file=uploaded_file,
            source_text=source_text,
            parser_mode=parser_mode,
            page_limit=int(page_limit),
        )
        state = paper.to_state()
        metadata = paper.metadata
        status = (
            f"Processed with **{paper.parser_mode}** mode in "
            f"**{metadata['conversion_seconds']} seconds**. "
            f"Extracted **{metadata['word_count']} words** across "
            f"**{metadata['section_count']} sections**."
        )
        return state, paper.markdown, section_table(paper.sections), status
    except Exception as exc:  # Gradio callback boundary
        traceback.print_exc()
        return {}, "", [], f"Processing failed: {_friendly_error(exc)}"


@gpu_task
def handle_summary(state, technicality_label, summarizer_label):
    try:
        technicality = technicality_label.lower()
        mode = "pegasus" if summarizer_label.startswith("PEGASUS") else "extractive"
        state, summary = summarize_state(state or {}, technicality, mode)
        return state, summary, f"Summary generated with **{mode}** mode."
    except Exception as exc:
        traceback.print_exc()
        return state or {}, "", f"Summary failed: {_friendly_error(exc)}"


def handle_narration(state):
    try:
        state, narration = narration_from_state(state or {})
        return state, narration, "Narration script prepared."
    except Exception as exc:
        traceback.print_exc()
        return state or {}, "", f"Narration failed: {_friendly_error(exc)}"


@gpu_task
def handle_audio(state, voice, speed):
    try:
        path = audio_from_state(state or {}, voice=voice, speed=float(speed))
        return path, "Audio generated with Kokoro-82M."
    except Exception as exc:
        traceback.print_exc()
        return None, f"Audio failed: {_friendly_error(exc)}"



with gr.Blocks(title="PaperCast") as demo:
    state = gr.State({})
    gr.Markdown(
        "# PaperCast \n"
        "Convert scientific PDFs with Granite-Docling, summarize them, and generate an audio briefing."
    )

    with gr.Tab("1. Process paper"):
        with gr.Row():
            uploaded_file = gr.File(label="Upload a PDF", file_types=[".pdf"], type="filepath")
            source_text = gr.Textbox(
                label="Or enter an arXiv ID / PDF URL",
                placeholder="2501.17887 or https://arxiv.org/pdf/2501.17887",
            )
        with gr.Row():
            parser_label = gr.Radio(
                ["Granite VLM", "Standard Docling"],
                value="Granite VLM",
                label="Parser",
            )
            page_limit = gr.Slider(1, 12, value=3, step=1, label="Pages to process")
        process_button = gr.Button("Process paper", variant="primary")
        process_status = gr.Markdown()
        with gr.Row():
            markdown_output = gr.Markdown(label="Structured output")
            section_output = gr.Dataframe(
                headers=["#", "Level", "Section", "Words"],
                datatype=["number", "number", "str", "number"],
                interactive=False,
                label="Detected sections",
            )
        process_button.click(
            handle_process,
            inputs=[uploaded_file, source_text, parser_label, page_limit],
            outputs=[state, markdown_output, section_output, process_status],
        )

    with gr.Tab("2. Summarize"):
        with gr.Row():
            technicality = gr.Radio(
                ["Overview", "Intermediate", "Technical"],
                value="Intermediate",
                label="Technicality",
            )
            summarizer_label = gr.Radio(
                ["Extractive baseline (fast)", "PEGASUS-X"],
                value="Extractive baseline (fast)",
                label="Summarizer",
            )
        summary_button = gr.Button("Generate summary", variant="primary")
        summary_status = gr.Markdown()
        summary_output = gr.Textbox(label="Summary", lines=18)
        summary_button.click(
            handle_summary,
            inputs=[state, technicality, summarizer_label],
            outputs=[state, summary_output, summary_status],
        )

    with gr.Tab("3. Audio"):
        narration_button = gr.Button("Prepare narration script")
        narration_status = gr.Markdown()
        narration_output = gr.Textbox(label="Speech-friendly script", lines=15)
        narration_button.click(
            handle_narration,
            inputs=[state],
            outputs=[state, narration_output, narration_status],
        )

        with gr.Row():
            voice = gr.Dropdown(
                ["af_heart", "af_bella", "am_adam", "am_michael"],
                value="af_heart",
                label="Kokoro voice",
            )
            speed = gr.Slider(0.8, 1.25, value=1.0, step=0.05, label="Speech speed")
        audio_button = gr.Button("Generate audio", variant="primary")
        audio_status = gr.Markdown()
        audio_output = gr.Audio(label="PaperCast audio", type="filepath")
        audio_button.click(
            handle_audio,
            inputs=[state, voice, speed],
            outputs=[audio_output, audio_status],
        )


if __name__ == "__main__":
    demo.queue().launch()
