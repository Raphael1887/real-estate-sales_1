1. Titre et description du projet

Ce projet a pour objectif d’analyser et de modéliser les ventes immobilières dans l’État du Connecticut (USA) entre 2001 et 2023, à partir de données publiques issues de data.ct.gov

Le pipeline ETL (Extraction, Transformation, Loading) automatise :
- Le téléchargement des données brutes
- Le nettoyage et la transformation des variables
- L’exploration statistique
- L’entraînement d’un modèle de régression linéaire pour prédire le prix d’un bien immobilier
- La mise à disposition d’une API prédictive déployable sur Azure


2. Technologies utilisées
- Python 3.11-alpine
- Pandas, NumPy, Scikit-learn - aujouter ce qu'on retrouve sur requirement
- Flask / FastAPI (pour l’API) --  à vérifier
- Azure App Service (déploiement cloud) --  à vérifier
- Git & GitHub / Azure Repos --  à vérifier


3. Prérequis
Avant de lancer le projet, assurez-vous d’avoir :
- Python ≥ 3.10 installé
- Git installé
- Une connexion Internet pour télécharger les données
- Un compte Azure (si déploiement souhaité)


4. Installation & Utilisation
Étape 1 – Cloner le dépôt
cd real-estate-etl
git clone https://github.com/Raphael1887/real-estate-sales_1.git

Étape 2 Télécharger les données
Le script télécharge les données brutes et crée le fichier nettoyé :
cd scripts
python3 download_data.py
Les données sont sauvegardées dans : data/housing_data.csv

Étape 3 Entraîner le modèle
Lancer le pipeline complet (chargement, exploration, entraînement, évaluation) :
cd scripts
python3 etl_pipeline.py

Étape 4 - Lancer L'api
cd real-estate-etl
chmod +x start_api.sh
./start_api.sh

Le code permet : 
- Créer et activer l’environnement virtuel
- Installer les dépendances
- Lancer l'API

Puis accéder à : http://localhost:8080

Étape 5 - Deploiement sur Azure

