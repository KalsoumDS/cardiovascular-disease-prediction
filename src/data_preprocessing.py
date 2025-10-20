import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Charge les données depuis le fichier CSV"""
    return pd.read_csv(file_path)

def visualize_data(df):
    """Visualise les données avec différents graphiques"""
    # Matrice de corrélation
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Matrice de Corrélation')
    plt.savefig('correlation_matrix.png')
    plt.close()

    # Distribution des variables numériques
    numeric_cols = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, hue='target', kde=True)
        plt.title(f'Distribution de {col} par classe')
        plt.savefig(f'distribution_{col}.png')
        plt.close()

def preprocess_data(df, is_training=False):
    """Prétraite les données"""
    # Copie du dataframe pour éviter les modifications sur l'original
    df_processed = df.copy()
    
    # Définition des catégories possibles pour chaque variable catégorielle
    categories = {
        'chest pain type': [1, 2, 3, 4],
        'resting ecg': [0, 1, 2],
        'ST slope': [1, 2, 3]
    }
    
    # Création des variables dummy avec toutes les catégories possibles
    for col, cats in categories.items():
        # Création d'un DataFrame temporaire avec les colonnes dummy
        dummy_df = pd.get_dummies(df_processed[col], prefix=col)
        
        # S'assurer que toutes les catégories sont présentes
        for cat in cats:
            col_name = f"{col}_{cat}"
            if col_name not in dummy_df.columns:
                dummy_df[col_name] = 0
        
        # Réorganiser les colonnes dans le même ordre que lors de l'entraînement
        dummy_cols = [f"{col}_{cat}" for cat in cats]
        dummy_df = dummy_df.reindex(columns=dummy_cols, fill_value=0)
        
        # Ajouter les colonnes dummy au DataFrame principal
        df_processed = pd.concat([df_processed, dummy_df], axis=1)
        
        # Supprimer la colonne originale
        df_processed.drop(col, axis=1, inplace=True)
    
    # Normalisation des variables numériques
    numeric_cols = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak']
    
    if is_training:
        # En phase d'entraînement, on ajuste le scaler et on le sauvegarde
        scaler = StandardScaler()
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
        # Sauvegarder les paramètres du scaler
        pd.DataFrame(
            {
                'mean': scaler.mean_,
                'scale': scaler.scale_
            },
            index=numeric_cols
        ).to_csv('scaler_params.csv')
    else:
        # En phase de prédiction, on utilise les paramètres sauvegardés
        scaler_params = pd.read_csv('scaler_params.csv', index_col=0)
        for col in numeric_cols:
            df_processed[col] = (df_processed[col] - scaler_params.loc[col, 'mean']) / scaler_params.loc[col, 'scale']
    
    # Réorganiser les colonnes dans le même ordre que lors de l'entraînement
    if not is_training:
        expected_columns = pd.read_csv('feature_columns.csv')['0'].tolist()
        df_processed = df_processed.reindex(columns=expected_columns, fill_value=0)
    
    return df_processed

def apply_pca(X, n_components=None):
    """Applique l'analyse en composantes principales"""
    if n_components is None:
        pca = PCA()
        pca.fit(X)
        # Trouver le nombre optimal de composantes
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= 0.95) + 1
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Visualisation de la variance expliquée
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Nombre de composantes')
    plt.ylabel('Variance expliquée cumulative')
    plt.title('Variance expliquée par les composantes principales')
    plt.savefig('pca_variance.png')
    plt.close()
    
    return X_pca, pca

def split_data(df, target_col='target', test_size=0.2, random_state=42):
    """Divise les données en ensembles d'entraînement et de test"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    # Chargement des données
    df = load_data('Base de donnée ML.csv')
    
    # Visualisation des données
    visualize_data(df)
    
    # Prétraitement avec is_training=True pour l'entraînement initial
    df_processed = preprocess_data(df, is_training=True)
    
    # Division des données
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    # Application de PCA
    X_train_pca, pca = apply_pca(X_train)
    X_test_pca = pca.transform(X_test)
    
    print("Forme des données d'entraînement:", X_train.shape)
    print("Forme des données de test:", X_test.shape)
    print("Forme des données PCA d'entraînement:", X_train_pca.shape)
    print("Forme des données PCA de test:", X_test_pca.shape) 