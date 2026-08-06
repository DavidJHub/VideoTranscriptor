"""Video transcription with two interchangeable approaches.

* ``deepgram`` - the hosted Deepgram API. Fast, needs a key and a network.
* ``canary``   - NVIDIA Canary 1B through NeMo, running on your own hardware.

Both consume the same normalised audio, so their outputs are directly
comparable; see :mod:`videotranscriptor.compare`.
"""

from .models import Segment, Transcript, TranscriptionError

__version__ = "0.1.0"

__all__ = ["Segment", "Transcript", "TranscriptionError", "__version__"]
