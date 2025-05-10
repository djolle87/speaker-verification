import logging
import os
from uuid import uuid1

import numpy as np
import torchaudio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from speechbrain.inference.classifiers import EncoderClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('speechbrain').setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def get_qdrant_host():
    """
    Determines if the script is running inside Docker or on the host machine,
    and returns the appropriate Qdrant host.
    """
    if os.path.exists('/.dockerenv'):
        # Inside Docker container, use the container name 'qdrant'
        return 'qdrant'
    else:
        # On localhost, use 'localhost'
        return 'localhost'


class SpeakerVerifierQdrant:
    """
    A speaker verification system using SpeechBrain for speaker embedding extraction
    and Qdrant as a vector database for storing and querying speaker embeddings.

    Attributes:
        classifier (EncoderClassifier): Pretrained SpeechBrain speaker encoder.
        qdrant (QdrantClient): Qdrant vector DB client instance.
    """

    def __init__(self, host=None, port=6333):
        """
        Initializes the verifier with a pretrained speaker encoder and connects to Qdrant.

        Args:
            host (str): Host address of the Qdrant server (defaults to None to auto-detect).
            port (int): Port of the Qdrant server.
        """
        logger.info("Initializing SpeakerVerifierQdrant...")

        # Auto-detect host if not provided
        if host is None:
            host = get_qdrant_host()

        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb"
        )
        logger.info("Pretrained classifier loaded.")

        self.qdrant = QdrantClient(host=host, port=port, grpc_port=port)
        self.collection_name = "speaker_verification"
        self.vectors_config = VectorParams(size=512, distance=Distance.COSINE)

        if not self.qdrant.collection_exists(collection_name=self.collection_name):
            logger.info(f"Collection '{self.collection_name}' not found. Creating new collection.")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.vectors_config
            )
        else:
            logger.info(f"Collection '{self.collection_name}' already exists.")

        logger.info("SpeakerVerifierQdrant initialized successfully.")

    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extracts a speaker embedding from a WAV audio file.

        Args:
            audio_path (str): Path to the WAV file.

        Returns:
            np.ndarray: A 512-dimensional speaker embedding.
        """
        logger.info(f"Extracting embedding from audio file: {audio_path}")
        signal, fs = torchaudio.load(audio_path)
        embedding = self.classifier.encode_batch(signal)
        logger.info("Embedding extraction complete.")
        return embedding[0][0].detach().numpy().astype(np.float32)

    def enroll(self, audio_path: str, speaker_name: str) -> str:
        """
        Enrolls a new speaker by extracting and storing their voice embedding.
        Prevents duplicate speaker names.

        Args:
            audio_path (str): Path to the speaker's WAV file.
            speaker_name (str): Unique name identifier for the speaker.

        Returns:
            str: Confirmation or error message.
        """
        logger.info(f"Enrolling speaker: {speaker_name} from audio file: {audio_path}")
        existing = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=self.extract_embedding(audio_path).tolist(),
            query_filter=Filter(
                must=[FieldCondition(key="name", match=MatchValue(value=speaker_name))]
            )
        )

        if existing:
            logger.warning(f"Speaker '{speaker_name}' is already enrolled.")
            return f"⚠️ Speaker '{speaker_name}' is already enrolled."

        embedding = self.extract_embedding(audio_path)
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=str(uuid1()),
                vector=embedding.tolist(),
                payload={"name": speaker_name}
            )]
        )

        info = self.qdrant.get_collection(collection_name=self.collection_name)
        logger.info(f"Speaker '{speaker_name}' enrolled. Currently enrolled speakers: {info.points_count}")
        return f"✅ Speaker '{speaker_name}' enrolled. Currently enrolled speakers: {info.points_count}"

    def verify(self, audio_path: str, claimed_name: str, threshold=0.016) -> str:
        """
        Verifies a speaker by comparing the audio against the stored embedding of the claimed name.

        Args:
            audio_path (str): Path to the speaker's WAV file.
            claimed_name (str): The name of the speaker the user claims to be.
            threshold (float): Similarity threshold (cosine) for verification success.

        Returns:
            str: Verification result message.
        """
        logger.info(f"Verifying speaker: {claimed_name} using audio file: {audio_path}")
        embedding = self.extract_embedding(audio_path)

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=embedding.tolist(),
            limit=10,
        )

        logger.info(f"Search results: {results}")

        if not results:
            logger.warning(f"No embedding found for '{claimed_name}'.")
            return f"❌ No embedding found for '{claimed_name}'."

        score = results.points[0].score
        if claimed_name == results.points[0].payload['name']:
            logger.info(f"Verification successful for '{claimed_name}' (similarity={score:.4f})")
            return f"✅ Verified as '{claimed_name}' (similarity={score:.4f})"
        else:
            logger.warning(f"Verification failed for '{claimed_name}' (similarity={score:.4f})")
            return f"❌ Verification failed for '{claimed_name}'"
