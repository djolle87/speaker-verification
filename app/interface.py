import logging

import gradio as gr
import pandas as pd

from app.verifier import SpeakerVerifierQdrant

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

verifier = SpeakerVerifierQdrant()
COLLECTION_NAME = verifier.collection_name


def enroll_ui(audio_path, name):
    logger.info(f"Starting enrollment for speaker: {name}, audio file: {audio_path}")
    msg = verifier.enroll(audio_path, name)
    logger.info(f"Enrollment completed for speaker: {name}")

    scroll_result = verifier.qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )
    points = scroll_result[0]
    data = [{"Speaker ID": point.id, "Speaker Name": point.payload['name']} for point in points]
    df = pd.DataFrame(data)
    logger.info(f"Latest entries fetched from collection: {len(points)} entries")
    return msg, df


def verify_ui(audio_path, name):
    logger.info(f"Verifying speaker: {name}, audio file: {audio_path}")
    result = verifier.verify(audio_path, name)
    logger.info(f"Verification result for {name}: {result}")
    return result


def clear_collection():
    logger.info(f"Clearing collection: {COLLECTION_NAME}")

    verifier.qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=verifier.vectors_config,
    )
    logger.info(f"Collection {COLLECTION_NAME} cleared.")
    return "🧹 Collection cleared (all vectors removed)."


with gr.Blocks() as demo:
    gr.Markdown("### Speaker Verification System")

    with gr.Tabs():
        with gr.Tab("Enroll"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(type="filepath", label="Audio")
                    name_input = gr.Text(label="Name")
                    enroll_btn = gr.Button("Enroll Speaker")

                with gr.Column(scale=1):
                    status_output = gr.Text(label="Enrollment Status")
                    table_output = gr.Dataframe(
                        headers=["Speaker ID", "Speaker Name"],
                        datatype=["str", "str"],
                        label="Vector DB (Latest Entries)"
                    )
                    clear_btn = gr.Button("🧹 Clear Vector DB")

            enroll_btn.click(fn=enroll_ui, inputs=[audio_input, name_input], outputs=[status_output, table_output])
            clear_btn.click(fn=clear_collection, outputs=status_output)

        with gr.Tab("Verify"):
            verify_audio = gr.Audio(type="filepath", label="Audio")
            verify_name = gr.Text(label="Claimed Name")
            verify_btn = gr.Button("Verify Speaker")
            verify_result = gr.Text(label="Verification Result")
            verify_btn.click(fn=verify_ui, inputs=[verify_audio, verify_name], outputs=verify_result)
