import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================
#     FONCTION 1 : LOAD & CLEAN
# ======================

def load_and_clean_data(csv_path):
    """
    Charge le fichier CSV, nettoie les données :
      - supprime les doublons
      - gère les valeurs manquantes
      - vérifie les types de données
    """
    print("📂 Chargement des données...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"✅ {len(df):,} lignes chargées")

    # Normaliser les noms de colonnes
    df.columns = df.columns.str.lower().str.strip()

    # Vérification des colonnes attendues
    expected_cols = ["surface", "chambres", "age_bien", "quartier_score", "distance_centre", "prix"]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Colonnes manquantes : {missing_cols}")

    # Supprimer les doublons
    df = df.drop_duplicates()

    # Gestion des valeurs manquantes
    missing_before = df.isna().sum().sum()
    df = df.dropna(subset=["prix"], how="any")
    df = df.fillna(df.median(numeric_only=True))
    missing_after = df.isna().sum().sum()

    print(f"🧹 Valeurs manquantes avant : {missing_before} → après : {missing_after}")

    # Convertir les types en numériques si besoin
    numeric_cols = ["surface", "chambres", "age_bien", "quartier_score", "distance_centre", "prix"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("✅ Données nettoyées avec succès")
    return df


# ======================
#     FONCTION 2 : EXPLORE DATA
# ======================

def explore_data(df):
    """
    Explore les données :
      - affiche les statistiques descriptives
      - calcule les corrélations
      - identifie les outliers
    """
    print("\n🔍 Exploration des données")

    print("\n📊 Statistiques descriptives :")
    print(df.describe().transpose())

    print("\n📈 Corrélations (top 5 avec le prix) :")
    corr = df.corr(numeric_only=True)["prix"].sort_values(ascending=False).head(5)
    print(corr)

    # Détection des outliers sur la variable cible
    q1 = df["prix"].quantile(0.25)
    q3 = df["prix"].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df["prix"] < q1 - 1.5 * iqr) | (df["prix"] > q3 + 1.5 * iqr)]
    print(f"\n⚠️ Outliers détectés : {len(outliers):,} lignes ({len(outliers)/len(df)*100:.2f}%)")


# ======================
#     FONCTION 3 : TRAIN MODEL
# ======================

def train_model(df):
    """
    Entraîne un modèle Random Forest pour prédire le prix.

    -----
    4.1 Comprendre le modèle choisi
      • Algorithme : Random Forest Regressor
      ✓ Robuste aux outliers
      ✓ Gère bien les relations non-linéaires
      ✓ Peu de prétraitement nécessaire
      ✓ Fournit l’importance des features
      ✓ Bonnes performances générales

    -----
    4.2 Hyperparamètres choisis :
      • n_estimators=100     → Nombre d'arbres
      • max_depth=20         → Profondeur maximale
      • min_samples_split=5  → Échantillons min pour split
      • random_state=42      → Reproductibilité
    """
    print("\n🤖 Entraînement du modèle Random Forest...")

    features = ["surface", "chambres", "age_bien", "quartier_score", "distance_centre"]
    X = df[features]
    y = df["prix"]

    # 1. Séparation features / target
    # 2. Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Entraînement du modèle
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print(f"✅ Modèle Random Forest entraîné avec {len(X_train):,} échantillons")

    # 6. Sauvegarde du modèle
    model_path = os.path.join(os.path.dirname(__file__), "..", "api", "model", "housing_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"💾 Modèle sauvegardé dans : {model_path}")

    return model, X_test, y_test


# ======================
#     FONCTION 4 : TEST MODEL
# ======================

def test_model_predictions(model, X_test, y_test):
    """
    Évalue le modèle sur l’échantillon de test et affiche les métriques.
    """
    print("\n🧪 Évaluation du modèle...")

    # 4. Prédictions sur test set
    y_pred = model.predict(X_test)

    # 5. Calcul des métriques
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"📉 MAE  : {mae:,.2f}")
    print(f"📉 RMSE : {rmse:,.2f}")
    print(f"📈 R²   : {r2:.3f}")

    # ======================
    #     TESTER DES PRÉDICTIONS MANUELLES
    # ======================
    print("\n🧮 Tester des prédictions manuelles (5 exemples)")
    print("   Valeur réelle vs valeur prédite + erreurs")

    sample_df = X_test.copy()
    sample_df["prix_reel"] = y_test
    sample_df["prix_pred"] = y_pred
    sample_df["erreur_absolue"] = abs(sample_df["prix_reel"] - sample_df["prix_pred"])
    sample_df["erreur_relative_%"] = (sample_df["erreur_absolue"] / sample_df["prix_reel"]) * 100

    print(sample_df[["prix_reel", "prix_pred", "erreur_absolue", "erreur_relative_%"]]
          .head(5)
          .round(2)
          .to_string(index=False))

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ======================
#     MAIN PIPELINE
# ======================

def main():
    """Pipeline complet ETL + entraînement modèle Random Forest."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "housing_data.csv")

    # Étape 1 : Extraction & nettoyage
    df = load_and_clean_data(data_path)

    # Étape 2 : Exploration
    explore_data(df)

    # Étape 3 : Entraînement du modèle
    model, X_test, y_test = train_model(df)

    # Étape 4 : Évaluation et test de prédictions
    test_model_predictions(model, X_test, y_test)

    print("\n🎉 Pipeline ETL terminé avec succès !")


if __name__ == "__main__":
    main()