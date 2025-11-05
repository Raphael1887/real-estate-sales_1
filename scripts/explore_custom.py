import pandas as pd
import matplotlib.pyplot as plt
import os

# === Chargement des données ===
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "housing_data.csv")
df = pd.read_csv(data_path)

# === Statistiques descriptives ===
print("=== STATISTIQUES DESCRIPTIVES ===")
print(df.describe())

# === Corrélations avec le prix ===
# Vérifie que la colonne "saleamount" existe (dans certains cas, elle s'appelle "prix" ou "price")
target_col = "saleamount" if "saleamount" in df.columns else "prix"

print("\n=== CORRÉLATIONS AVEC LE PRIX ===")
corr = df.corr(numeric_only=True)[target_col].sort_values(ascending=False)
print(corr)

# === Visualisations (optionnel) ===
# Génération d'histogrammes des variables numériques
print("\n📊 Génération des histogrammes...")
df.hist(figsize=(15, 10))
plt.tight_layout()

# Sauvegarde dans le dossier data/
output_path = os.path.join(os.path.dirname(__file__), "..", "data", "histograms.png")
plt.savefig(output_path)
plt.close()
print(f"✅ Histogrammes sauvegardés dans {output_path}")