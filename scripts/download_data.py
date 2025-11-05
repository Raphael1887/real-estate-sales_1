import pandas as pd
import os
import numpy as np
from datetime import datetime
 
URL = "https://data.ct.gov/resource/5mzw-sjtu.csv?$limit=500000"
 
OUTPUT_DIR = "./data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "housing_data.csv")
 
def main():
    print("📥 Téléchargement des données...")
    df = pd.read_csv(URL)
    print(f"✅ Données téléchargées : {len(df):,} lignes")
 
    # Nettoyage de base
    df = df.drop_duplicates().dropna(how="all")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
 
    # --- Création des 6 colonnes du TP ---
 
    # 1️⃣ Surface : simulation entre 500 et 4000 sqft
    np.random.seed(42)
    df["surface"] = np.random.randint(500, 10000, size=len(df))
 
    # 2️⃣ Chambres : fonction de la surface
    df["chambres"] = np.random.randint(1, 8, size=len(df))
 
    # 3️⃣ Age du bien : simulation entre 0 et 100 ans
    df["age_bien"] = np.random.randint(0, 150, size=len(df))
 
    # 4️⃣ Quartier_score : Note de 1 à 10 aléatoirement
    df["quartier_score"] = np.random.randint(1.0, 10.0, size=len(df))
 
    # 5️⃣ Distance au centre : simulation
    df["distance_centre"] = np.random.randint(0, 20, size=len(df))
 
    # 6️⃣ Prix : utiliser saleamount
    df["prix"] = pd.to_numeric(df.get("saleamount"), errors="coerce").fillna(100000)
 
    # Sélection finale des colonnes
    df_final = df[["surface", "chambres", "age_bien", "quartier_score", "distance_centre", "prix"]]
 
    # Sauvegarde
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Fichier final sauvegardé : {OUTPUT_FILE}")
    print("✅ Aperçu :")
    print(df_final.head())
 
if __name__ == "__main__":
    main()