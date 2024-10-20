from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import pickle
import numpy as np
from http import HTTPStatus
import logging
import warnings
import os
from pathlib import Path
from datetime import datetime

# Configuration du logging
log_directory = Path('logs')
log_directory.mkdir(exist_ok=True)
log_file = log_directory / f'credit_prediction_api_{datetime.now().strftime("%Y%m%d")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Configuration de l'application Flask
app = Flask(__name__, template_folder=Path(__file__).parent / 'templates')
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=1800,  # 30 minutes
    MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max-limit
)

# Configuration CORS sécurisée
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:8000", "http://127.0.0.1:8000"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Variables globales
REQUIRED_FEATURES = [
    'credit_lines_outstanding',
    'loan_amt_outstanding',
    'total_debt_outstanding',
    'income',
    'years_employed',
    'fico_score'
]

# Limites pour la validation des données
FEATURE_LIMITS = {
    'credit_lines_outstanding': {'min': 0, 'max': 100},
    'loan_amt_outstanding': {'min': 0, 'max': 10000000},
    'total_debt_outstanding': {'min': 0, 'max': 10000000},
    'income': {'min': 0, 'max': 10000000},
    'years_employed': {'min': 0, 'max': 100},
    'fico_score': {'min': 300, 'max': 850}
}

model = None

def get_model_path():
    """Retourne le chemin du modèle en tenant compte de l'environnement"""
    current_dir = Path(__file__).resolve().parent
    model_path = current_dir.parent / 'random_forest_model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(f"Le modèle n'existe pas à l'emplacement: {model_path}")
    return model_path

def load_model():
    """Charge le modèle Random Forest depuis le fichier pickle."""
    global model
    if model is not None:
        return model
    logger.info("Tentative de chargement du modèle...")
    try:
        model_path = get_model_path()
        logger.info(f"Tentative de chargement depuis: {model_path}")
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        logger.info("Modèle chargé avec succès!")
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        raise

def validate_feature_value(feature, value):
    """Valide une valeur pour une feature donnée"""
    limits = FEATURE_LIMITS[feature]
    if not (limits['min'] <= value <= limits['max']):
        raise ValueError(
            f"La valeur {value} pour {feature} est hors limites. "
            f"Min: {limits['min']}, Max: {limits['max']}"
        )

def validate_input_data(data):
    """Valide les données d'entrée et retourne les features formatées"""
    if not data:
        raise ValueError("Aucune donnée n'a été fournie")
    missing_features = [feature for feature in REQUIRED_FEATURES if feature not in data]
    if missing_features:
        raise ValueError(f"Features manquantes: {missing_features}")
    features = []
    for feature in REQUIRED_FEATURES:
        try:
            value = float(data[feature])
            if not np.isfinite(value):
                raise ValueError(f"La valeur pour {feature} n'est pas un nombre valide")
            validate_feature_value(feature, value)
            features.append(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Erreur de validation pour {feature}: {str(e)}")
    return np.array(features).reshape(1, -1)

@app.route('/')
def home():
    """Route racine fournissant des informations sur l'API"""
    try:
        model_path = get_model_path()
        model_status = "loaded" if model_path.exists() else "not found"
    except Exception:
        model_status = "error"
    return jsonify({
        "message": "Bienvenue sur l'API de prédiction de défaut de crédit",
        "status": "healthy",
        "model_status": model_status,
        "endpoints": {
            "/": {"method": "GET", "description": "Page d'accueil avec documentation"},
            "/ui": {"method": "GET", "description": "Interface utilisateur web"},
            "/predict": {
                "method": "POST",
                "description": "Faire une prédiction de défaut de crédit",
                "required_features": REQUIRED_FEATURES,
                "feature_limits": FEATURE_LIMITS,
                "example_payload": {feature: limits['min'] for feature, limits in FEATURE_LIMITS.items()}
            }
        }
    })

@app.route('/ui')
def ui():
    """Route pour servir l'interface utilisateur"""
    try:
        ui_path = Path(__file__).parent / 'templates' / 'index.html'
        with open(ui_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return render_template_string(content)
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'interface: {str(e)}")
        return jsonify({
            "error": "Interface utilisateur non trouvée",
            "detail": str(e)
        }), HTTPStatus.NOT_FOUND

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour effectuer des prédictions."""
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.info(f"Nouvelle requête de prédiction - ID: {request_id}")
    try:
        data = request.json
        logger.debug(f"Données reçues - ID {request_id}: {data}")
        features = validate_input_data(data)
        model = load_model()
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        response = {
            "request_id": request_id,
            "prediction": int(prediction),
            "probability": {
                "default": float(probabilities[1]),
                "no_default": float(probabilities[0])
            },
            "features_received": dict(zip(REQUIRED_FEATURES, features[0])),
            "interpretation": "Risque de défaut détecté" if prediction == 1 else "Pas de risque de défaut détecté"
        }
        logger.info(f"Prédiction réussie - ID {request_id}: {response['interpretation']}")
        return jsonify(response), HTTPStatus.OK
    except ValueError as e:
        error_msg = f"Erreur de validation - ID {request_id}: {str(e)}"
        logger.warning(error_msg)
        return jsonify({
            "error": "Erreur de validation des données",
            "request_id": request_id,
            "detail": str(e)
        }), HTTPStatus.BAD_REQUEST
    except Exception as e:
        error_msg = f"Erreur interne - ID {request_id}: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "error": "Erreur lors du traitement de la prédiction",
            "request_id": request_id,
            "detail": str(e) if app.debug else "Une erreur interne s'est produite"
        }), HTTPStatus.INTERNAL_SERVER_ERROR

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Route non trouvée",
        "available_endpoints": ["/", "/predict", "/ui"]
    }), HTTPStatus.NOT_FOUND

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "error": "Erreur interne du serveur",
        "detail": str(e) if app.debug else "Une erreur interne s'est produite"
    }), HTTPStatus.INTERNAL_SERVER_ERROR

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Erreur non gérée: {str(e)}")
    return jsonify({
        "error": "Erreur inattendue",
        "detail": str(e) if app.debug else "Une erreur inattendue s'est produite"
    }), HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    logger.warning("Exécution en mode développement. Utilisez Waitress pour la production.")
    try:
        load_model()
    except Exception as e:
        logger.warning(f"Attention: Le modèle n'a pas pu être chargé au démarrage: {str(e)}")
    app.run(debug=False, port=8000, host='0.0.0.0')