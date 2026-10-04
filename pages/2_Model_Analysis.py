import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

st.set_page_config(page_title="Analyse des Données et des Modèles", page_icon="", layout="wide")

# Chargement des données et des résultats
try:
    results_df = pd.read_csv('reports/model_results.csv')
    df = pd.read_csv('data/data.csv')
    
    # Création d'onglets pour séparer l'analyse des données et l'analyse des modèles
    tab1, tab2 = st.tabs([" Analyse des Relations", " Performance des Modèles"])
    
    with tab1:
        st.title("Analyse des Relations avec la Variable Cible")
        
        # Section 1: Distribution de la Variable Cible
        st.header("1. Distribution de la Variable Cible")
        col1, col2 = st.columns(2)
        
        with col1:
            target_counts = df['target'].value_counts()
            st.metric("Nombre de cas sans maladie", target_counts[0])
            st.metric("Nombre de cas avec maladie", target_counts[1])
        
        with col2:
            target_dist = df['target'].value_counts(normalize=True)
            fig_target = px.pie(values=target_dist.values, 
                              names=["Pas de maladie", "Maladie cardiaque"],
                              title="Distribution des Classes")
            st.plotly_chart(fig_target)
        
        # Section 2: Corrélations avec la Variable Cible
        st.header("2. Corrélations avec la Variable Cible")
        corr_matrix = df.corr()
        target_corr = corr_matrix['target'].sort_values(ascending=False)
        fig_corr = px.bar(x=target_corr.index, y=target_corr.values,
                         title="Force de la Relation entre Chaque Variable et la Maladie Cardiaque",
                         labels={'x': 'Variables', 'y': 'Coefficient de Corrélation'})
        fig_corr.update_layout(showlegend=False)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Interprétation spécifique
        st.subheader("Interprétation spécifique des corrélations avec la variable cible")
        st.markdown("""
- La variable la plus corrélée avec la maladie est `ST slope`, suivie de `chest pain type`, `exercise angina` et `oldpeak`.
- Les autres variables (`sex`, `age`, `fasting blood sugar`, etc.) ont une corrélation beaucoup plus faible avec la maladie.
- Cela signifie que les variables liées à l'électrocardiogramme et à la douleur thoracique sont les plus informatives pour prédire la maladie cardiaque dans cette base.
- Les variables comme le cholestérol ou la fréquence cardiaque maximale, bien que médicalement pertinentes, n'apportent pas ici une forte capacité de discrimination dans ce jeu de données.
""")
        
        # Section 3: Analyse Détaillée des Relations
        st.header("3. Analyse Détaillée des Relations")
        
        # Sélection de variable
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        selected_var = st.selectbox("Sélectionnez une variable à analyser", 
                                  [col for col in numeric_cols if col != 'target'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot
            fig_box = px.box(df, x='target', y=selected_var, 
                           title=f"Distribution de {selected_var} par classe",
                           labels={'target': 'Maladie Cardiaque (0: Non, 1: Oui)', 
                                 'y': selected_var})
            st.plotly_chart(fig_box)
            
            # Statistiques descriptives
            st.subheader("Statistiques Descriptives")
            stats_df = df.groupby('target')[selected_var].describe()
            stats_df.index = ['Sans Maladie', 'Avec Maladie']
            st.dataframe(stats_df)
            
            # Interprétation spécifique pour le box plot
            if selected_var == "age":
                st.markdown("""
- La distribution de l'âge est très similaire pour les deux classes (maladie présente ou non).
- Les médianes et les étendues sont proches.
- L'âge seul ne permet pas de distinguer efficacement les deux groupes dans cette base.
""")
            elif selected_var == "sex":
                st.markdown("""
- Les deux sexes sont représentés, mais il y a plus d'hommes (1) que de femmes (0) dans la base.
- La maladie cardiaque touche les deux sexes, sans différence majeure visible dans la distribution.
""")
        
        with col2:
            # Histogramme
            fig_hist = px.histogram(df, x=selected_var, color='target',
                                  title=f"Distribution de {selected_var} selon la présence de maladie",
                                  labels={'target': 'Maladie Cardiaque',
                                        'count': 'Nombre de cas'},
                                  color_discrete_map={0: 'blue', 1: 'red'},
                                  barmode='overlay')
            fig_hist.update_layout(showlegend=True)
            st.plotly_chart(fig_hist)
            
            # Test statistique
            group0 = df[df['target'] == 0][selected_var]
            group1 = df[df['target'] == 1][selected_var]
            stat, p_value = np.mean(group0), np.mean(group1)
            st.metric("Différence des moyennes", f"{stat - p_value:.2f}")
    
    with tab2:
        st.title("Analyse des Performances des Modèles")
        
        # Section 1: Vue d'ensemble des performances
        st.header("1. Comparaison Globale des Modèles")
        metrics = st.multiselect(
            "Sélectionnez les métriques à comparer",
            ['Accuracy', 'Precision', 'Recall'],
            default=['Accuracy']
        )
        
        if metrics:
            fig = px.bar(results_df, 
                        x='Modèle', 
                        y=metrics,
                        barmode='group',
                        title="Comparaison des Performances des Modèles")
            st.plotly_chart(fig, use_container_width=True)
        
        # Section 2: Analyse détaillée par modèle
        st.header("2. Analyse Détaillée par Modèle")
        model_choice = st.selectbox(
            "Sélectionnez un modèle à analyser",
            results_df['Modèle'].tolist()
        )
        
        if model_choice:
            model_name = model_choice.lower().replace(" ", "_")
            
            # Métriques principales
            st.subheader("Métriques de Performance")
            model_metrics = results_df[results_df['Modèle'] == model_choice].iloc[0]
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{model_metrics['Accuracy']:.3f}")
                st.metric("Precision", f"{model_metrics['Precision']:.3f}")
            
            with col2:
                st.metric("Recall", f"{model_metrics['Recall']:.3f}")
            
            # Interprétation des métriques pour la régression logistique
            if model_choice == "Régression Logistique":
                st.markdown("""
**Interprétation des métriques** :
- Le modèle prédit correctement 87 % des cas (accuracy).
- Parmi les cas prédits comme "maladie", 86,8 % sont réellement malades (précision).
- Il détecte 90,1 % des vrais cas de maladie (recall).
""")
            
            # Interprétation des métriques pour la random forest
            if model_choice == "Random Forest":
                st.markdown("""
**Interprétation des métriques** :
- Le modèle prédit correctement 95 % des cas (accuracy), ce qui est excellent.
- Parmi les cas prédits comme "maladie", 94,1 % sont réellement malades (précision).
- Il détecte 96,9 % des vrais cas de maladie (recall).
""")
            
            # Visualisations
            col1, col2 = st.columns(2)
            
            with col1:
                confusion_matrix_path = f'confusion_matrix_{model_name}.png'
                if os.path.exists(confusion_matrix_path):
                    st.subheader("Matrice de Confusion")
                    st.image(confusion_matrix_path)
                    # Interprétation matrice de confusion
                    if model_choice == "Random Forest":
                        st.markdown("""
**Interprétation** :
- 99 vrais négatifs (prédits sains, réellement sains)
- 127 vrais positifs (prédits malades, réellement malades)
- 8 faux positifs (prédits malades, mais réellement sains)
- 4 faux négatifs (prédits sains, mais réellement malades)
- Le modèle fait très peu d'erreurs, surtout sur les cas de maladie (seulement 4 faux négatifs).
""")
                
                roc_curve_path = f'roc_curve_{model_name}.png'
                if os.path.exists(roc_curve_path):
                    st.subheader("Courbe ROC")
                    st.image(roc_curve_path)
                    # Interprétation courbe ROC
                    if model_choice == "Random Forest":
                        st.markdown("""
**Interprétation** :
- L'aire sous la courbe (AUC = 0.97) est excellente.
- Le modèle distingue très bien les malades des non malades.
""")
            
            with col2:
                pr_curve_path = f'precision_recall_{model_name}.png'
                if os.path.exists(pr_curve_path):
                    st.subheader("Courbe Précision-Rappel")
                    st.image(pr_curve_path)
                    # Interprétation courbe précision-rappel
                    if model_choice == "Random Forest":
                        st.markdown("""
**Interprétation** :
- La courbe reste très élevée, ce qui montre que le modèle garde une excellente précision même quand il cherche à détecter un maximum de malades.
- Le modèle est performant pour minimiser à la fois les faux positifs et les faux négatifs.
""")
                
                learning_curve_path = f'learning_curve_{model_name}.png'
                if os.path.exists(learning_curve_path):
                    st.subheader("Courbe d'Apprentissage")
                    st.image(learning_curve_path)
                    # Interprétation courbe d'apprentissage
                    if model_choice == "Random Forest":
                        st.markdown("""
**Interprétation** :
- Les scores d'entraînement et de validation sont très proches, ce qui montre que le modèle n'est pas en sur-apprentissage.
- Le score de validation augmente avec la taille de l'échantillon, ce qui indique que le modèle bénéficie de plus de données.
""")
            
            # Importance des caractéristiques pour les modèles appropriés
            if model_choice in ['Arbre de Décision', 'Random Forest']:
                feature_importance_path = f'feature_importance_{model_name}.png'
                if os.path.exists(feature_importance_path):
                    st.header("Importance des Caractéristiques")
                    st.image(feature_importance_path)
                    # Interprétation importance des caractéristiques
                    if model_choice == "Random Forest":
                        st.markdown("""
**Interprétation** :
- Les variables les plus importantes pour la prédiction sont `ST slope`, `oldpeak`, `max heart rate`, `chest pain type`, et `cholesterol`.
- Cela confirme l'importance des variables liées à l'électrocardiogramme et à la douleur thoracique dans la prédiction de la maladie cardiaque.
""")

except FileNotFoundError:
    st.error("Les fichiers nécessaires ne sont pas disponibles. Veuillez d'abord exécuter l'entraînement des modèles.") 