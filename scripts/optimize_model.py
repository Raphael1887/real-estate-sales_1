import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ======================
#     OPTIMISATION RANDOM FOREST
# ======================

def main():
    print("📂 Chargement des données...")
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "housing_data.csv")
    df = pd.read_csv(data_path)

    # Vérification basique
    if "prix" not in df.columns:
        raise KeyError("La colonne cible 'prix' est manquante dans le dataset.")

    # Séparation features / target
    X = df.drop("prix", axis=1)
    y = df["prix"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🔍 Lancement du Grid Search pour Random Forest...")

    # Grille d’hyperparamètres
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5, 10]
    }

    # Modèle de base
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Grid Search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Résultats du tuning
    print("\n🏆 Meilleurs paramètres trouvés :")
    print(grid_search.best_params_)
    print(f"⭐ Meilleur score R² (CV) : {grid_search.best_score_:.3f}")

    # Évaluation sur le test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Évaluation sur le jeu de test :")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R²   : {r2:.3f}")

    # Sauvegarde du modèle optimisé
    model_dir = os.path.join(os.path.dirname(__file__), "..", "api", "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "housing_model_optimized.pkl")

    joblib.dump(best_model, model_path)
    print(f"\n💾 Modèle sauvegardé dans : {model_path}")


if __name__ == "__main__":
    main()