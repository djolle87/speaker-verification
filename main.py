import logging

from app.interface import demo

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('speechbrain').setLevel(logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {e}")
