# src/api/ensemble.py
import asyncio
from typing import Dict, Any
from src.core.interfaces import IRiskAnalyzer
from src.core.logger import get_logger

logger = get_logger(__name__)

class EnsembleScorer:
    def __init__(self, vision: IRiskAnalyzer, price: IRiskAnalyzer, text: IRiskAnalyzer):
        self.vision = vision
        self.price = price
        self.text = text

    async def calculate_risk(self, listing_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Triggering concurrent ML analyzers (Vision, Price, Text)...")
        
        vision_task = self.vision.analyze(listing_data)
        price_task = self.price.analyze(listing_data)
        text_task = self.text.analyze(listing_data)
        
        vision_res, price_res, text_res = await asyncio.gather(vision_task, price_task, text_task)
        
        v_score, v_proof = vision_res
        p_score, p_proof = price_res
        t_score, t_proof = text_res

        final_score = (0.40 * v_score) + (0.35 * p_score) + (0.25 * t_score)
        logger.info(f"Ensemble execution complete. Component Scores -> Vision: {v_score}, Price: {p_score}, Text: {t_score}")
        logger.info(f"Final calculated Scam Score: {final_score:.1f}/100")
        
        if final_score < 40:
            risk_level = "Safe"
            action = "Standard diligence. Safe to visit."
        elif final_score < 70:
            risk_level = "Suspicious"
            action = "Proceed with caution. Verify photos match reality."
        else:
            risk_level = "High Risk"
            action = "High Risk - Verify owner ID + electricity bill before paying any token."

        return {
            "scam_score": round(final_score, 1),
            "risk_level": risk_level,
            "action_required": action,
            "proofs": [
                {"category": "Vision", "description": v_proof},
                {"category": "Price", "description": p_proof},
                {"category": "Language", "description": t_proof}
            ]
        }