import os
import joblib
import logging
import numpy as np
from flask import Flask, jsonify, request
from datetime import datetime
from functools import lru_cache
from flask_swagger_ui import get_swaggerui_blueprint

# ============================================================
# 1️⃣ CONFIGURATION
# ============================================================
class Config:
    MODEL_PATH = os.getenv('MODEL_PATH', 'model/housing_model.pkl')
    MODEL_VERSION = os.getenv('MODEL_VERSION', '1.0.0')
    PORT = int(os.getenv('PORT', 8080))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

    # Limites pour validation
    SURFACE_MIN = 0
    SURFACE_MAX = 1000
    CHAMBRES_MIN = 0
    CHAMBRES_MAX = 10


# ============================================================
# 2️⃣ LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 3️⃣ CHARGEMENT DU MODÈLE
# ============================================================
model = None

def load_model():
    global model
    try:
        model = joblib.load(Config.MODEL_PATH)
        logger.info("✅ Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle : {e}")
        model = None


# ============================================================
# 4️⃣ INITIALISATION FLASK
# ============================================================
app = Flask(__name__)
logger.info("🚀 Démarrage de l'API...")
load_model()


# ============================================================
# 5️⃣ VALIDATION D’ENTRÉES
# ============================================================
def validate_input(data):
    """Vérifie que les entrées sont valides avant prédiction."""
    required_fields = ["surface", "chambres", "age_bien", "quartier_score", "distance_centre"]

    for field in required_fields:
        if field not in data:
            return False, f"Champ manquant : '{field}'"

    try:
        float(data["surface"])
        int(data["chambres"])
        float(data["age_bien"])
        float(data["quartier_score"])
        float(data["distance_centre"])
    except ValueError:
        return False, "Types de données invalides"

    if not (Config.SURFACE_MIN <= float(data["surface"]) <= Config.SURFACE_MAX):
        return False, f"Surface hors limites ({Config.SURFACE_MIN}-{Config.SURFACE_MAX})"
    if not (Config.CHAMBRES_MIN <= int(data["chambres"]) <= Config.CHAMBRES_MAX):
        return False, f"Nombre de chambres hors limites ({Config.CHAMBRES_MIN}-{Config.CHAMBRES_MAX})"

    return True, None


# ============================================================
# 6️⃣ CACHING DES PRÉDICTIONS
# ============================================================
@lru_cache(maxsize=100)
def predict_cached(surface, chambres, age_bien, quartier_score, distance_centre):
    """Cache les prédictions pour améliorer les performances"""
    features = np.array([[surface, chambres, age_bien, quartier_score, distance_centre]])
    return float(model.predict(features)[0])


# ============================================================
# 7️⃣ ENDPOINTS
# ============================================================

@app.route("/", methods=["GET"])
def root():
    """Endpoint racine : infos générales"""
    return jsonify({
        "message": "API Housing Price Prediction",
        "version": Config.MODEL_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "predict_batch": "/predict/batch (POST)",
            "model_info": "/model/info",
            "metrics": "/metrics",
            "docs": "/docs"
        },
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/health", methods=["GET"])
def health():
    """Vérifie la santé du modèle"""
    status = "healthy" if model is not None else "degraded"
    return jsonify({
        "status": status,
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "version": Config.MODEL_VERSION
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Prédiction pour un seul bien."""
    if model is None:
        return jsonify({"error": "Le modèle n’est pas chargé."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Aucune donnée reçue."}), 400

    is_valid, error_msg = validate_input(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    try:
        prix = predict_cached(
            float(data["surface"]),
            int(data["chambres"]),
            float(data["age_bien"]),
            float(data["quartier_score"]),
            float(data["distance_centre"])
        )

        logger.info(f"✅ Prédiction réussie : {prix}")

        return jsonify({
            "prix_estime": round(prix, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": Config.MODEL_VERSION,
            "input": data
        })

    except Exception as e:
        logger.error(f"Erreur de prédiction : {e}")
        return jsonify({"error": "Erreur interne de prédiction."}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Prédire pour plusieurs biens à la fois"""
    try:
        data = request.get_json()
        if not isinstance(data, list):
            return jsonify({'error': 'Attendu : liste de biens'}), 400

        predictions = []
        for item in data:
            is_valid, error_msg = validate_input(item)
            if not is_valid:
                return jsonify({'error': error_msg}), 400

            prix = predict_cached(
                float(item["surface"]),
                int(item["chambres"]),
                float(item["age_bien"]),
                float(item["quartier_score"]),
                float(item["distance_centre"])
            )

            predictions.append({
                'input': item,
                'prix_estime': round(prix, 2)
            })

        return jsonify({
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Erreur batch : {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# 8️⃣ SWAGGER UI (Documentation)
# ============================================================
SWAGGER_URL = '/docs'
API_URL = '/static/swagger.json'  # Fichier swagger.json à placer dans /static/
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


# ============================================================
# 9️⃣ LANCEMENT
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)