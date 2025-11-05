import os
import joblib
import numpy as np

# ======================
#     TEST DU MODÈLE
# ======================

def main():
    print("📦 Chargement du modèle Random Forest...")

    # Chemin du modèle
    model_path = os.path.join(os.path.dirname(__file__), "..", "api", "model", "housing_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modèle introuvable : {model_path}\n"
                                "➡️  Assurez-vous d’avoir exécuté etl_pipeline.py avant ce test.")

    # Charger le modèle entraîné
    model = joblib.load(model_path)
    print("✅ Modèle chargé avec succès !")

    # ======================
    #     TESTS DE PRÉDICTIONS
    # ======================

    # Rappel des features : [surface, chambres, age_bien, quartier_score, distance_centre]

    test_samples = {
        "Petit appartement": np.array([[45, 1, 5, 6, 3.5]]),
        "Grande maison": np.array([[150, 4, 20, 9, 1.2]]),
        "Bien ancien": np.array([[80, 2, 50, 5, 8.0]])
    }

    print("\n🏠 Tests de prédiction sur 3 biens :\n")

    for label, features in test_samples.items():
        prediction = model.predict(features)[0]
        print(f"🔹 {label:<20} → Prix estimé : {prediction:,.0f} €")

    print("\n❓ Les prédictions sont-elles cohérentes avec la réalité ?")


if __name__ == "__main__":
    main()