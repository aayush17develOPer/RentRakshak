# src/tabular/analyzer.py
import json
import joblib
from typing import Dict, Any, Tuple
from src.core.interfaces import IRiskAnalyzer
from src.tabular.train_mock import PriceFeatureEngineer

class PriceAnomalyAnalyzer(IRiskAnalyzer):
    """
    Implements the Price Defense Layer.
    Detects "Too Good to be True" listings using RF regression residuals.
    """
    
    def __init__(self, model_path: str = "models/price_model.joblib", medians_path: str = "models/sector_median.json"):
        try:
            self.model = joblib.load(model_path)
            with open(medians_path, "r") as f:
                self.sector_medians = json.load(f)
            self.engineer = PriceFeatureEngineer(self.sector_medians)
        except Exception as e:
            print(f"Warning: Failed to load price ML artifacts. Run train_mock.py first. Error: {e}")
            self.model = None

    async def analyze(self, listing_data: Dict[str, Any]) -> Tuple[float, str]:
        if not self.model:
            return 0.0, "Price anomaly skipped: Model not loaded."
            
        actual_price = listing_data.get("price")
        if not actual_price:
            return 0.0, "No price provided in listing."

        try:
            # 1. Engineer Features
            features = self.engineer.engineer_features(listing_data)
            
            # 2. Predict Fair Market Value
            predicted_price = self.model.predict([features])[0]
            
            # 3. Calculate Residual Percentage
            # Formula: (actual - predicted) / predicted
            residual_pct = (actual_price - predicted_price) / predicted_price
            
            # 4. Evaluate Anomaly Thresholds
            if residual_pct < -0.35:
                # Actual is > 35% cheaper than expected for these amenities
                score = abs(residual_pct) * 100
                score = min(score, 100.0) # Cap at 100
                
                furnishing = listing_data.get("furnishing_status", "unfurnished")
                gated = "gated" if listing_data.get("gated") == 1 else "non-gated"
                bhk = listing_data.get("bhk", 1)
                sector = listing_data.get("sector", "unknown sector")
                
                proof = (f"Price {abs(residual_pct)*100:.0f}% below expected. "
                         f"For a {furnishing} {gated} {bhk} BHK in {sector}, "
                         f"expected ~₹{predicted_price:,.0f}, listed at ₹{actual_price:,.0f}.")
                return score, proof
                
            elif residual_pct > 0.50:
                return 0.0, "Listing is heavily overpriced, but does not fit standard bait-and-switch scam profile."
                
            return 0.0, f"Price is reasonable for given amenities (Expected: ~₹{predicted_price:,.0f})."
            
        except Exception as e:
            print(f"Price anomaly analysis failed: {e}")
            return 0.0, "Price analysis skipped due to processing error."

# --- Quick Test Block ---
if __name__ == "__main__":
    import asyncio
    
    async def test_price_anomaly():
        analyzer = PriceAnomalyAnalyzer()
        
        # Scenario 1: Normal Listing
        normal_listing = {
            "price": 9500, "bhk": 1, "area_sqft": 180, "floor": "Ground", 
            "furnishing_status": "unfurnished", "age": "5-10 Years", 
            "gated": 0, "amenities_count": 1, "sector": "Cartarpuri"
        }
        
        # Scenario 2: Scam Listing (Too good to be true for luxury amenities)
        scam_listing = {
            "price": 8000, "bhk": 1, "area_sqft": 250, "floor": "3", 
            "furnishing_status": "fully", "age": "new", 
            "gated": 1, "amenities_count": 4, "sector": "Cartarpuri"
        }
        
        print("Evaluating Normal Listing...")
        s1, p1 = await analyzer.analyze(normal_listing)
        print(f"Score: {s1:.1f}, Proof: {p1}\n")
        
        print("Evaluating Luxury Scam Listing...")
        s2, p2 = await analyzer.analyze(scam_listing)
        print(f"Score: {s2:.1f}, Proof: {p2}")

    asyncio.run(test_price_anomaly())