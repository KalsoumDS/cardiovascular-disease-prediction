import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Charger les données existantes
df_existing = pd.read_csv('Base de donnée ML.csv')

# Calculer les statistiques pour chaque colonne
stats = df_existing.describe()

# Générer 7310 nouvelles lignes (pour atteindre 8500 au total)
n_new_rows = 7310

# Créer un nouveau DataFrame avec des données générées
new_data = pd.DataFrame()

# Générer des données pour chaque colonne en respectant les distributions
for column in df_existing.columns:
    if column == 'oldpeak':  # Colonne float
        mean = stats[column]['mean']
        std = stats[column]['std']
        new_data[column] = np.random.normal(mean, std, n_new_rows)
    else:  # Colonnes entières
        min_val = stats[column]['min']
        max_val = stats[column]['max']
        mean = stats[column]['mean']
        std = stats[column]['std']
        # Générer des données avec une distribution normale tronquée
        new_data[column] = np.random.normal(mean, std, n_new_rows)
        new_data[column] = np.clip(new_data[column], min_val, max_val)
        new_data[column] = new_data[column].round().astype(int)

# Combiner les données existantes avec les nouvelles
combined_df = pd.concat([df_existing, new_data], ignore_index=True)

# Sauvegarder le nouveau fichier
combined_df.to_csv('Base de donnée ML.csv', index=False)

print(f"Données générées avec succès. Nouveau nombre total d'observations : {len(combined_df)}") 