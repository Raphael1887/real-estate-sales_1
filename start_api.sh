#!/bin/bash

echo "🚀 Démarrage de l'API Flask..."

# Aller dans le dossier de l’API
cd "$(dirname "$0")/api" || exit 1

# Activer l'environnement virtuel
if [ -d "../env" ]; then
    source ../env/bin/activate
    echo "✅ Environnement virtuel activé."
else
    echo "⚠️  Aucun environnement virtuel trouvé (../env)"
    echo "Création d’un nouvel environnement..."
    python3 -m venv ../env
    source ../env/bin/activate
    pip install -r requirements.txt
fi

# Lancer l’API Flask
echo "🌐 Lancement du serveur Flask..."
python3 app.py