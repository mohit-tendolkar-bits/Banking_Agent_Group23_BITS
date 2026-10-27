"""Browser UI to test the assistant. Each message runs main.handle(); the tool calls are shown next to the reply.
Start with `uv run ui.py` and open http://127.0.0.1:7860."""

import contextlib
import io

import gradio as gr

from main import handle

EXAMPLES = [
    "I lost my card yesterday, please block it",
    "How long does a card take to arrive?",
    "Can you cancel the transfer I just made?",
    "A payment I made was declined, why?",
    "What is the exchange rate you use?",
]


def respond(message, history, pending):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):  # main.call_tool prints one "[tool] ..." line per call
        reply, pending = handle(message, pending)
    trace = out.getvalue().strip().replace(" -> ", "\n  -> ").replace("  [tool]", "\n[tool]")
    if pending:
        trace += f"\n\n[waiting for confirmation] {pending}"
    return reply, trace, pending


with gr.Blocks(title="Banking assistant") as demo:
    gr.Markdown("# Banking assistant\nActions that change the account (block a card, cancel a transfer, ...) are confirmed before they run.")
    with gr.Row():
        with gr.Column(scale=2):
            trace = gr.Code(label="Tool calls (last message)", language=None, lines=14, wrap_lines=True, show_line_numbers=False, render=False)
            pending = gr.State(None)
            gr.ChatInterface(respond, examples=[[e] for e in EXAMPLES], additional_inputs=[pending], additional_outputs=[trace, pending])
        with gr.Column(scale=1):
            trace.render()

if __name__ == "__main__":
    demo.launch()
