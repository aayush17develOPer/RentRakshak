# src/nlp/analyzer.py
import os
import numpy as np
from typing import Dict, Any, Tuple
from src.core.interfaces import IRiskAnalyzer

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    HAS_NLP_LIBS = True
except ImportError:
    HAS_NLP_LIBS = False
    print("Warning: transformers and onnxruntime are required for full NLP analysis.")

class TextScamAnalyzer(IRiskAnalyzer):
    """
    Implements the Text Defense Layer.
    Detects high-pressure Hinglish scam language using IndicBERT via ONNX runtime[cite: 9].
    """
    
    def __init__(self, 
                 model_path: str = "models/text_classifier.onnx", 
                 tokenizer_name: str = "ai4bharat/indic-bert"):
        
        self.model_path = model_path
        self.is_ready = False
        
        if HAS_NLP_LIBS and os.path.exists(self.model_path):
            # Load the production ONNX model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.session = ort.InferenceSession(self.model_path)
            self.is_ready = True
        else:
            print("Note: ONNX model not found. Using Mock Text Classifier for LLD testing.")

    def _mock_predict(self, text: str) -> Tuple[int, float]:
        """A simple keyword fallback so the pipeline works before the model is trained."""
        text = text.lower()
        scam_keywords = ["token", "advance", "turant", "booking confirm", "direct owner"]
        
        match_count = sum(1 for word in scam_keywords if word in text)
        if match_count >= 2:
            return 2, 0.95  # advance_demand
        elif match_count == 1:
            return 1, 0.70  # high_pressure
        return 0, 0.10      # normal

    async def analyze(self, listing_data: Dict[str, Any]) -> Tuple[float, str]:
        description = listing_data.get("description", "")
        if not description:
            return 0.0, "No text description provided."

        try:
            if self.is_ready:
                # --- Production ONNX Inference ---
                inputs = self.tokenizer(
                    description, 
                    return_tensors="np", 
                    truncation=True, 
                    max_length=128, 
                    padding="max_length"
                )
                
                # ONNX expects specific input names (usually input_ids, attention_mask)
                onnx_inputs = {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                }
                
                logits = self.session.run(None, onnx_inputs)[0]
                probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                
                class_id = int(np.argmax(probabilities, axis=1)[0])
                confidence = float(probabilities[0][class_id])
                
            else:
                # --- Development Mock Inference ---
                class_id, confidence = self._mock_predict(description)
            
            # Map labels as per roadmap: 0=normal, 1=high_pressure, 2=advance_demand[cite: 9]
            if class_id == 2:
                score = confidence * 100
                proof = "High-risk demand for advance payment detected in description."
                return min(score, 100.0), proof
                
            elif class_id == 1:
                score = confidence * 100
                proof = "High-pressure urgency language detected (e.g., 'turant', 'abhi')."
                return min(score, 100.0), proof
                
            return 0.0, "Description language appears normal and low-pressure."
            
        except Exception as e:
            print(f"Text analysis failed: {e}")
            return 0.0, "Text analysis skipped due to processing error."


# --- Quick Test Block ---
if __name__ == "__main__":
    import asyncio
    
    async def test_text_anomaly():
        analyzer = TextScamAnalyzer()
        
        normal_listing = {
            "description": "Standard 1 RK available for rent. Family preferred. 1 month deposit."
        }
        
        scam_listing = {
            "description": "Bhai flat available hai, bahut demand hai turant token advance se booking confirm karo direct owner"
        }
        
        print("Evaluating Normal Text...")
        s1, p1 = await analyzer.analyze(normal_listing)
        print(f"Score: {s1:.1f}, Proof: {p1}\n")
        
        print("Evaluating Scam Text...")
        s2, p2 = await analyzer.analyze(scam_listing)
        print(f"Score: {s2:.1f}, Proof: {p2}")

    asyncio.run(test_text_anomaly())