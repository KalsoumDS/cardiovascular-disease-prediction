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

st.set_page_config(page_title="Analyse des données et des modèles", page_icon="", layout="wide")

# Chargement des données et des résultats
try:
    results_df = pd.read_csv('reports/model_results.csv')
    df = pd.read_csv('data/data.csv')
    
    # Création d'onglets pour séparer l'analyse des données et l'analyse des modèles
    tab1, tab2 = st.tabs(["Analyse des relations", "Performance des modèles"])
    
    with tab1:
        st.title("Analyse des relations avec la variable cible")
        
        # Section 1: Distribution de la Variable Cible
        st.header("1. Répartition de la variable cible")
        col1, col2 = st.columns(2)
        
        with col1:
            target_counts = df['target'].value_counts()
            st.metric("Cas sans maladie", target_counts[0])
            st.metric("Cas avec maladie", target_counts[1])
        
        with col2:
            target_dist = df['target'].value_counts(normalize=True)
            fig_target = px.pie(values=target_dist.values, 
                              names=["Absence de maladie", "Maladie cardiovasculaire"],
                              title="Répartition des classes")
            st.plotly_chart(fig_target)
        
        # Section 2: Corrélations avec la Variable Cible
        st.header("2. Corrélations avec la variable cible")
        feature_labels = {
            "age": "Âge",
            "sex": "Sexe",
            "chest pain type": "Type de douleur thoracique",
            "resting bp s": "Pression artérielle au repos",
            "cholesterol": "Cholestérol",
            "fasting blood sugar": "Glycémie à jeun",
            "resting ecg": "Électrocardiogramme au repos",
            "max heart rate": "Fréquence cardiaque maximale",
            "exercise angina": "Angine d’effort",
            "oldpeak": "Dépression du segment ST",
            "ST slope": "Pente du segment ST",
        }
        corr_matrix = df.corr()
        target_corr = corr_matrix['target'].sort_values(ascending=False)
        target_corr.index = [
            feature_labels.get(col, "Maladie cardiovasculaire" if col == "target" else col)
            for col in target_corr.index
        ]
        fig_corr = px.bar(x=target_corr.index, y=target_corr.values,
                         title="Lien entre chaque variable et la maladie cardiovasculaire",
                         labels={'x': 'Variables', 'y': 'Coefficient de corrélation'})
        fig_corr.update_layout(showlegend=False)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Interprétation spécifique
        st.subheader("Interprétation des corrélations avec la variable cible")
        st.markdown("""
- Les variables les plus corrélées avec la maladie sont la pente du segment ST, le type de douleur thoracique, l’angine d’effort et la dépression du segment ST.
- Les autres variables (sexe, âge, glycémie à jeun, etc.) ont une corrélation beaucoup plus faible avec la maladie.
- Cela signifie que les variables liées à l’électrocardiogramme et à la douleur thoracique sont les plus informatives pour prédire la maladie cardiovasculaire dans cette base.
- Le cholestérol et la fréquence cardiaque maximale, bien que médicalement pertinents, distinguent peu les classes dans ce jeu de données.
""")
        
        # Section 3: Analyse Détaillée des Relations
        st.header("3. Analyse détaillée des relations")
        
        # Sélection de variable
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        selected_var = st.selectbox(
            "Sélectionnez une variable à analyser",
            [col for col in numeric_cols if col != 'target'],
            format_func=lambda col: feature_labels.get(col, col),
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot
            df_plot = df.assign(Statut=df["target"].map({0: "Sans maladie", 1: "Maladie cardiovasculaire"}))
            fig_box = px.box(df_plot, x='Statut', y=selected_var,
                           title=f"Répartition de {feature_labels.get(selected_var, selected_var)} par classe",
                           labels={'Statut': 'Statut clinique',
                                 'y': feature_labels.get(selected_var, selected_var)})
            st.plotly_chart(fig_box)
            
            # Statistiques descriptives
            st.subheader("Statistiques descriptives")
            stats_df = df.groupby('target')[selected_var].describe()
            stats_df.index = ['Sans maladie', 'Avec maladie']
            stats_df = stats_df.rename(columns={
                "count": "Effectif", "mean": "Moyenne", "std": "Écart-type",
                "min": "Minimum", "25%": "1er quartile", "50%": "Médiane",
                "75%": "3e quartile", "max": "Maximum",
            })
            stats_df.columns.name = feature_labels.get(selected_var, selected_var)
            st.dataframe(stats_df)
            
            # Interprétation spécifique pour le box plot
            if selected_var == "age":
                st.markdown("""
- La distribution de l’âge est très similaire pour les deux classes (maladie présente ou non).
- Les médianes et les étendues sont proches.
- L’âge seul ne permet pas de distinguer efficacement les deux groupes dans cette base.
""")
            elif selected_var == "sex":
                st.markdown("""
- Les deux sexes sont représentés, mais il y a plus d'hommes (1) que de femmes (0) dans la base.
- La maladie cardiaque touche les deux sexes, sans différence majeure visible dans la distribution.
""")
        
        with col2:
            # Histogramme
            fig_hist = px.histogram(df_plot, x=selected_var, color='Statut',
                                  title=f"Répartition de {feature_labels.get(selected_var, selected_var)} selon la présence de la maladie",
                                  labels={'Statut': 'Statut clinique',
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
        st.title("Analyse des performances des modèles")
        
        # Section 1: Vue d'ensemble des performances
        st.header("1. Comparaison globale des modèles")
        metric_labels = {
            "Accuracy": "Exactitude",
            "Precision": "Précision",
            "Recall": "Rappel",
        }
        metrics = st.multiselect(
            "Sélectionnez les métriques à comparer",
            ['Accuracy', 'Precision', 'Recall'],
            default=['Accuracy'],
            format_func=lambda metric: metric_labels[metric],
        )
        
        if metrics:
            fig = px.bar(results_df, 
                        x='Modèle', 
                        y=metrics,
                        barmode='group',
                        title="Comparaison des performances des modèles",
                        labels={**metric_labels, "Modèle": "Modèle", "value": "Score"})
            st.plotly_chart(fig, use_container_width=True)
        
        # Section 2: Analyse détaillée par modèle
        st.header("2. Analyse détaillée par modèle")
        model_choice = st.selectbox(
            "Sélectionnez un modèle à analyser",
            results_df['Modèle'].tolist(),
            format_func=lambda model: {
                "Régression Logistique": "Régression logistique",
                "Logistic Regression": "Régression logistique",
                "Random Forest": "Forêt aléatoire",
                "Decision Tree": "Arbre de décision",
                "Arbre de Décision": "Arbre de décision",
                "KNN": "K plus proches voisins (KNN)",
                "KMeans": "K-moyennes",
            }.get(model, model),
        )
        
        if model_choice:
            model_name = model_choice.lower().replace(" ", "_")
            
            # Métriques principales
            st.subheader("Indicateurs de performance")
            model_metrics = results_df[results_df['Modèle'] == model_choice].iloc[0]
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Exactitude", f"{model_metrics['Accuracy']:.3f}")
                st.metric("Précision", f"{model_metrics['Precision']:.3f}")
            
            with col2:
                st.metric("Rappel", f"{model_metrics['Recall']:.3f}")
            
            # Interprétation des métriques pour la régression logistique
            if model_choice == "Régression Logistique":
                st.markdown("""
**Interprétation des métriques** :
- Le modèle prédit correctement 87 % des cas (exactitude).
- Parmi les cas prédits comme "maladie", 86,8 % sont réellement malades (précision).
- Il détecte 90,1 % des vrais cas de maladie (rappel).
""")
            
            # Interprétation des métriques pour la random forest
            if model_choice == "Random Forest":
                st.markdown("""
**Interprétation des métriques** :
- Le modèle prédit correctement 95 % des cas (exactitude), ce qui est excellent.
- Parmi les cas prédits comme "maladie", 94,1 % sont réellement malades (précision).
- Il détecte 96,9 % des vrais cas de maladie (rappel).
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