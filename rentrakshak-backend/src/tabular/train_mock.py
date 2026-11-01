# src/tabular/train_mock.py
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os

class PriceFeatureEngineer:
    """Transforms raw scraper dictionary into numerical ML features."""
    
    def __init__(self, sector_medians: dict):
        self.sector_medians = sector_medians

    def engineer_features(self, listing: dict) -> list:
        # 1. Furnishing Score
        furnishing_map = {"unfurnished": 0, "semi": 1, "fully": 2}
        furnishing = furnishing_map.get(listing.get("furnishing_status", "").lower(), 0)
        
        # 2. Floor Score (Simplified mapping)
        floor_str = str(listing.get("floor", "0")).lower()
        if floor_str in ["ground", "0"]: floor_score = 0
        elif floor_str in ["1", "2"]: floor_score = 1
        elif floor_str.isdigit() and int(floor_str) > 5: floor_score = 3
        else: floor_score = 2
        
        # 3. Age Score
        age = listing.get("age", "new").lower()
        age_score = 2 if "new" in age else (1 if "5" in age else 0)
        
        # 4. Area per BHK
        bhk = listing.get("bhk", 1) or 1
        area_per_bhk = listing.get("area_sqft", 200) / bhk
        
        # 5. Sector Median (Target Encoding Fallback)
        sector = listing.get("sector", "")
        parent_sector = listing.get("parent_sector", "")
        median_price = self.sector_medians.get(sector, self.sector_medians.get(parent_sector, 10000))
        
        return [
            furnishing,
            floor_score,
            listing.get("gated", 0),
            listing.get("amenities_count", 0),
            area_per_bhk,
            median_price,
            age_score
        ]

def train_and_save_model():
    """Generates synthetic data and trains the RandomForestRegressor."""
    print("Training mock Random Forest model...")
    
    # 1. Save sector medians (Our Hierarchical Fallback)
    sector_medians = {"Cartarpuri": 9000, "Palam Vihar Ext": 14000, "HSR": 18000}
    with open("models/sector_median.json", "w") as f:
        json.dump(sector_medians, f)
        
    engineer = PriceFeatureEngineer(sector_medians)
    
    # 2. Generate Synthetic Training Data
    # [furnishing, floor, gated, amenities, area_per_bhk, median_price, age]
    X_train = [
        [0, 0, 0, 1, 180, 9000, 1],  # Normal Unfurnished
        [2, 2, 1, 4, 250, 9000, 2],  # Luxury Gated
        [1, 1, 0, 2, 200, 18000, 1], # Normal Semi HSR
    ]
    y_train = [8500, 16000, 17500]   # Expected reasonable prices
    
    # 3. Train Model
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Save Model
    joblib.dump(model, "models/price_model.joblib")
    print("Model saved to models/price_model.joblib")

if __name__ == "__main__":
    train_and_save_model()