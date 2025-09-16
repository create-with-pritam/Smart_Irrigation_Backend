# app.py - Flask backend for .pkl model
import os
import joblib
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Config ---
MODEL_PATH = os.environ.get("MODEL_PATH", "model/sprinkler_model.pkl")
PORT = int(os.environ.get("PORT", 8080))

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml_backend")

# --- Flask App ---
app = Flask(__name__)

# --- Load Model ---
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.exception(f"❌ Failed to load model from {MODEL_PATH}")
    model = None


# --- API Routes ---
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not available"}), 500

    try:
        data = request.get_json(force=True)

        # Expecting {"features": {...}} OR {"instances": [[...]]}
        if "instances" in data:
            instances = data["instances"]  # already a list of feature lists
        elif "features" in data:
            features = data["features"]   # dict of {feature_name: value}
            # Convert to list (order matters!)
            feature_order = [
                "season_Monsoon", "season_Post-Monsoon", "season_Pre-Monsoon", "season_Winter",
                "soil_moisture", "temperature", "humidity", "rain_probability",
                "time_of_day", "soil_ec"
            ]
            instance = [features.get(f, 0) for f in feature_order]
            instances = [instance]
        else:
            return jsonify({"error": "Invalid payload format"}), 400

        # Run prediction
        prediction = model.predict(instances).tolist()

        return jsonify({
            "status": "success",
            "prediction": prediction
        }), 200

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


# --- Main ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
