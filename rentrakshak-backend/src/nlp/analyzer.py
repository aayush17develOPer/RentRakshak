
import os
import time
import numpy as np
from typing import Dict, Any, Tuple

from src.core.interfaces import IRiskAnalyzer
from src.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------
# Dependency initialization
# ---------------------------------------------------------

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer

    HAS_NLP_LIBS = True
    logger.info(
        "NLP dependencies successfully imported | "
        "onnxruntime_version=%s | transformers_available=True",
        ort.__version__
    )

except ImportError:
    HAS_NLP_LIBS = False
    logger.exception(
        "NLP dependency import failed. "
        "Production NLP inference will be unavailable."
    )


class TextScamAnalyzer(IRiskAnalyzer):
    """
    Implements the Text Defense Layer.

    Detects high-pressure Hinglish scam language using
    IndicBERT via ONNX Runtime.

    Supports:
    - Production ONNX inference.
    - Keyword-based mock inference.
    - Detailed execution logging.
    """

    def __init__(
        self,
        model_path: str = "models/text_classifier.onnx",
        tokenizer_name: str = "ai4bharat/indic-bert"
    ):

        init_start = time.perf_counter()

        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.is_ready = False

        self.tokenizer = None
        self.session = None

        logger.info(
            "Initializing TextScamAnalyzer | "
            "model_path=%s | tokenizer=%s | "
            "nlp_dependencies_available=%s",
            self.model_path,
            self.tokenizer_name,
            HAS_NLP_LIBS
        )

        # Resolve model path relative to current working directory.
        absolute_model_path = os.path.abspath(self.model_path)

        logger.debug(
            "Resolved model path | absolute_path=%s | exists=%s",
            absolute_model_path,
            os.path.isfile(absolute_model_path)
        )

        if not HAS_NLP_LIBS:
            logger.warning(
                "Production NLP dependencies unavailable. "
                "Switching to mock classifier."
            )

        elif not os.path.isfile(absolute_model_path):
            logger.warning(
                "ONNX model file not found | path=%s | "
                "falling_back_to_mock=True",
                absolute_model_path
            )

        else:
            try:
                logger.info(
                    "Loading tokenizer | tokenizer_name=%s",
                    self.tokenizer_name
                )

                tokenizer_start = time.perf_counter()

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name
                )

                logger.info(
                    "Tokenizer loaded successfully | duration=%.3fs",
                    time.perf_counter() - tokenizer_start
                )

                logger.info(
                    "Creating ONNX inference session | model_path=%s",
                    absolute_model_path
                )

                session_start = time.perf_counter()

                self.session = ort.InferenceSession(
                    absolute_model_path
                )

                logger.info(
                    "ONNX session initialized | duration=%.3fs",
                    time.perf_counter() - session_start
                )

                # Inspect model input/output metadata.
                model_inputs = self.session.get_inputs()
                model_outputs = self.session.get_outputs()

                logger.info(
                    "ONNX model metadata | inputs=%s | outputs=%s",
                    [
                        {
                            "name": item.name,
                            "shape": item.shape,
                            "type": item.type
                        }
                        for item in model_inputs
                    ],
                    [
                        {
                            "name": item.name,
                            "shape": item.shape,
                            "type": item.type
                        }
                        for item in model_outputs
                    ]
                )

                self.is_ready = True

                logger.info(
                    "TextScamAnalyzer initialized successfully | "
                    "inference_mode=ONNX"
                )

            except Exception:
                self.is_ready = False

                logger.exception(
                    "Failed to initialize ONNX model or tokenizer. "
                    "Switching to mock inference."
                )

        logger.info(
            "TextScamAnalyzer initialization completed | "
            "mode=%s | duration=%.3fs",
            "ONNX" if self.is_ready else "MOCK",
            time.perf_counter() - init_start
        )

    # ---------------------------------------------------------
    # Mock inference
    # ---------------------------------------------------------

    def _mock_predict(self, text: str) -> Tuple[int, float]:
        """
        Keyword-based fallback classifier.
        """

        logger.debug(
            "Starting mock inference | text_length=%d",
            len(text)
        )

        scam_keywords = [
            "token",
            "advance",
            "turant",
            "booking confirm",
            "direct owner"
        ]

        normalized_text = text.lower()

        matched_keywords = [
            word for word in scam_keywords
            if word in normalized_text
        ]

        match_count = len(matched_keywords)

        logger.info(
            "Mock inference completed | "
            "matched_keywords=%s | match_count=%d",
            matched_keywords,
            match_count
        )

        if match_count >= 2:
            class_id, confidence = 2, 0.95

        elif match_count == 1:
            class_id, confidence = 1, 0.70

        else:
            class_id, confidence = 0, 0.10

        logger.info(
            "Mock prediction | class_id=%d | confidence=%.4f",
            class_id,
            confidence
        )

        return class_id, confidence

    # ---------------------------------------------------------
    # Main analysis pipeline
    # ---------------------------------------------------------

    async def analyze(
        self,
        listing_data: Dict[str, Any]
    ) -> Tuple[float, str]:

        analysis_start = time.perf_counter()

        logger.info(
            "Starting text scam analysis | "
            "inference_mode=%s",
            "ONNX" if self.is_ready else "MOCK"
        )

        try:
            # Step 1: Validate input.
            if not isinstance(listing_data, dict):
                logger.error(
                    "Invalid listing_data type | received_type=%s",
                    type(listing_data).__name__
                )
                return 0.0, "Invalid listing data provided."

            logger.debug(
                "Listing data received | available_keys=%s",
                list(listing_data.keys())
            )

            description = listing_data.get("description", "")

            if description is None:
                description = ""

            if not isinstance(description, str):
                logger.warning(
                    "Description is not a string | type=%s",
                    type(description).__name__
                )
                description = str(description)

            description = description.strip()

            logger.info(
                "Description extracted | "
                "description_length=%d | is_empty=%s",
                len(description),
                not bool(description)
            )

            if not description:
                logger.warning(
                    "Analysis skipped because listing description is empty."
                )
                return 0.0, "No text description provided."

            # Step 2: Select inference engine.
            if self.is_ready:

                logger.info(
                    "Executing production ONNX inference."
                )

                # Step 2A: Tokenization.
                tokenization_start = time.perf_counter()

                inputs = self.tokenizer(
                    description,
                    return_tensors="np",
                    truncation=True,
                    max_length=128,
                    padding="max_length"
                )

                logger.info(
                    "Tokenization completed | duration=%.3fs",
                    time.perf_counter() - tokenization_start
                )

                logger.debug(
                    "Tokenization metadata | %s",
                    {
                        key: {
                            "shape": value.shape,
                            "dtype": str(value.dtype)
                        }
                        for key, value in inputs.items()
                    }
                )

                # Step 2B: Prepare ONNX inputs.
                expected_inputs = self.session.get_inputs()

                logger.debug(
                    "Expected ONNX input names | names=%s",
                    [item.name for item in expected_inputs]
                )

                onnx_inputs = {
                    item.name: inputs[item.name]
                    for item in expected_inputs
                    if item.name in inputs
                }

                if len(onnx_inputs) != len(expected_inputs):
                    missing_inputs = [
                        item.name
                        for item in expected_inputs
                        if item.name not in onnx_inputs
                    ]

                    raise ValueError(
                        f"Missing required ONNX inputs: {missing_inputs}"
                    )

                logger.info(
                    "ONNX inputs prepared | input_names=%s",
                    list(onnx_inputs.keys())
                )

                # Step 2C: Run model inference.
                inference_start = time.perf_counter()

                logits = self.session.run(
                    None,
                    onnx_inputs
                )[0]

                inference_duration = (
                    time.perf_counter() - inference_start
                )

                logger.info(
                    "ONNX inference completed | "
                    "duration=%.3fs | logits_shape=%s",
                    inference_duration,
                    logits.shape
                )

                logger.debug(
                    "Raw model logits | logits=%s",
                    logits.tolist()
                )

                # Step 2D: Convert logits into probabilities.
                if logits.ndim != 2 or logits.shape[0] == 0:
                    raise ValueError(
                        f"Unexpected logits shape: {logits.shape}"
                    )

                # Numerically stable softmax.
                shifted_logits = logits - np.max(
                    logits,
                    axis=1,
                    keepdims=True
                )

                exp_logits = np.exp(shifted_logits)

                probabilities = exp_logits / np.sum(
                    exp_logits,
                    axis=1,
                    keepdims=True
                )

                class_id = int(
                    np.argmax(probabilities, axis=1)[0]
                )

                confidence = float(
                    probabilities[0][class_id]
                )

                logger.info(
                    "ONNX prediction generated | "
                    "class_id=%d | confidence=%.4f | "
                    "probabilities=%s",
                    class_id,
                    confidence,
                    probabilities[0].tolist()
                )

            else:

                # Step 2E: Mock inference.
                logger.warning(
                    "Using mock keyword classifier instead of ONNX model."
                )

                class_id, confidence = self._mock_predict(
                    description
                )

            # Step 3: Interpret prediction.
            logger.info(
                "Interpreting model prediction | "
                "class_id=%d | confidence=%.4f",
                class_id,
                confidence
            )

            if class_id == 2:

                score = min(confidence * 100, 100.0)

                proof = (
                    "High-risk demand for advance payment "
                    "detected in description."
                )

            elif class_id == 1:

                score = min(confidence * 100, 100.0)

                proof = (
                    "High-pressure urgency language detected "
                    "(e.g., 'turant', 'abhi')."
                )

            elif class_id == 0:

                score = 0.0

                proof = (
                    "Description language appears normal "
                    "and low-pressure."
                )

            else:
                logger.error(
                    "Unexpected class ID returned by model | "
                    "class_id=%d",
                    class_id
                )

                return (
                    0.0,
                    f"Unexpected classification label: {class_id}"
                )

            # Step 4: Final result.
            logger.info(
                "Text analysis completed | "
                "class_id=%d | score=%.2f | "
                "confidence=%.4f | duration=%.3fs",
                class_id,
                score,
                confidence,
                time.perf_counter() - analysis_start
            )

            return score, proof

        except Exception:

            logger.exception(
                "Text analysis failed | "
                "inference_mode=%s | description_length=%d",
                "ONNX" if self.is_ready else "MOCK",
                len(description) if "description" in locals() else -1
            )

            return (
                0.0,
                "Text analysis skipped due to processing error."
            )


# ---------------------------------------------------------
# Quick test block
# ---------------------------------------------------------

if __name__ == "__main__":

    import asyncio

    async def test_text_anomaly():

        logger.info("Starting standalone TextScamAnalyzer test.")

        analyzer = TextScamAnalyzer()

        normal_listing = {
            "description": (
                "Standard 1 RK available for rent. "
                "Family preferred. 1 month deposit."
            )
        }

        scam_listing = {
            "description": (
                "Bhai flat available hai, bahut demand hai "
                "turant token advance se booking confirm karo direct owner"
            )
        }

        logger.info("Testing normal listing.")

        s1, p1 = await analyzer.analyze(normal_listing)

        print(
            f"Normal listing | Score: {s1:.1f} | Proof: {p1}\n"
        )

        logger.info("Testing suspicious listing.")

        s2, p2 = await analyzer.analyze(scam_listing)

        print(
            f"Suspicious listing | Score: {s2:.1f} | Proof: {p2}"
        )

        logger.info("Standalone test completed.")

    asyncio.run(test_text_anomaly())
