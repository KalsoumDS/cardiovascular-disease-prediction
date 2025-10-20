import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_preprocessing import load_data, preprocess_data, split_data

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Visualise la matrice de confusion"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.ylabel('Valeurs Réelles')
    plt.xlabel('Prédictions')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name):
    """Visualise la courbe ROC"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title(f'Courbe ROC - {model_name}')
    plt.legend()
    plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, model_name):
    """Visualise la courbe Précision-Rappel"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Rappel')
    plt.ylabel('Précision')
    plt.title(f'Courbe Précision-Rappel - {model_name}')
    plt.savefig(f'precision_recall_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_learning_curve(model, X, y, model_name):
    """Visualise la courbe d'apprentissage"""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Score d\'entraînement')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, test_mean, label='Score de validation')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.xlabel('Taille de l\'ensemble d\'entraînement')
    plt.ylabel('Score')
    plt.title(f'Courbe d\'Apprentissage - {model_name}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'learning_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Entraîne et évalue différents modèles de machine learning"""
    models = {
        'Régression Logistique': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Arbre de Décision': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'KMeans': KMeans(n_clusters=2, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nEntraînement du modèle: {name}")
        
        if name == 'KMeans':
            # Pour KMeans, nous utilisons une approche différente
            model.fit(X_train)
            y_pred = model.predict(X_test)
            # Convertir les labels pour correspondre aux classes originales
            y_pred = np.where(y_pred == 0, 0, 1)
            y_prob = None
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = None
        
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Validation croisée
        if name != 'KMeans':
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        else:
            cv_mean = np.nan
            cv_std = np.nan
        
        # Visualisations
        plot_confusion_matrix(y_test, y_pred, name)
        if y_prob is not None:
            plot_roc_curve(y_test, y_prob, name)
            plot_precision_recall_curve(y_test, y_prob, name)
        if name != 'KMeans':
            plot_learning_curve(model, X_train, y_train, name)
        
        results.append({
            'Modèle': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'CV Mean': cv_mean,
            'CV Std': cv_std
        })
        
        # Sauvegarde du modèle
        joblib.dump(model, f'models/{name.lower().replace(" ", "_")}_model.joblib')
        
        # Visualisation des résultats pour les modèles appropriés
        if name in ['Arbre de Décision', 'Random Forest']:
            plot_feature_importance(model, X_train.columns, name)
    
    return pd.DataFrame(results)

def plot_feature_importance(model, feature_names, model_name):
    """Visualise l'importance des caractéristiques pour les modèles d'arbre"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Importance des caractéristiques - {model_name}')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), feature_names[indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

def plot_results(results_df):
    """Visualise les résultats des différents modèles"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        sns.barplot(x='Modèle', y=metric, data=results_df)
        plt.title(f'Comparaison des modèles - {metric}')
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

if __name__ == "__main__":
    # Chargement des données prétraitées
    from data_preprocessing import load_data, preprocess_data, split_data
    
    # Chargement et prétraitement des données
    df = load_data('Base de donnée ML.csv')
    df_processed = preprocess_data(df, is_training=True)
    X_train, X_test, y_train, y_test = split_data(df_processed)
    
    # Sauvegarder la liste des colonnes
    pd.Series(X_train.columns).to_csv('feature_columns.csv', index=False)
    
    # Entraînement et évaluation des modèles
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Affichage des résultats
    print("\nRésultats détaillés:")
    print(results)
    
    # Visualisation des résultats
    plot_results(results)
    
    # Sauvegarde des résultats
    results.to_csv('model_results.csv', index=False) 