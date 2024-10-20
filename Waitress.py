import os
import sys
from waitress import serve
import logging
from pathlib import Path

# Obtenir le chemin absolu du répertoire actuel
current_dir = Path(__file__).resolve().parent
api_dir = current_dir / 'credit_prediction_api'

# Ajouter le chemin au PYTHONPATH
sys.path.insert(0, str(api_dir))

try:
    from app import app
except ImportError as e:
    print(f"Erreur d'importation. Chemin de recherche Python actuel : {sys.path}")
    print(f"Erreur : {e}")
    print(f"Contenu du dossier {api_dir} :", os.listdir(api_dir) if api_dir.exists() else "Dossier non trouvé")
    sys.exit(1)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info(f"Démarrage du serveur depuis : {current_dir}")
    logger.info(f"Dossier API : {api_dir}")
    
    try:
        serve(
            app,
            host='0.0.0.0',
            port=8000,
            threads=4,
            channel_timeout=30,
            connection_limit=1000,
            cleanup_interval=30,
            url_scheme='http'
        )
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du serveur: {str(e)}")
        sys.exit(1)