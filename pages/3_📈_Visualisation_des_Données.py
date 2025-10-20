import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Exploration des Données", page_icon="📊")

st.title("Exploration et Analyse des Données")

try:
    # Chargement des données
    df = pd.read_csv('Base de donnée ML.csv')
    
    # Affichage des informations générales
    st.sidebar.header("Informations sur la Base de Données")
    st.sidebar.write(f"Nombre total d'observations : {len(df)}")
    st.sidebar.write(f"Nombre de variables : {len(df.columns)}")
    
    # Sélection du type de visualisation
    viz_type = st.selectbox(
        "Type de Visualisation",
        ["Aperçu des Données", "Matrice de Corrélation", 
         "Distribution des Variables", "Analyse Bivariée", 
         "Analyse PCA"]
    )
    
    if viz_type == "Aperçu des Données":
        st.header("Aperçu des Données")
        st.write("Aperçu des 5 premières lignes :")
        st.dataframe(df.head())
        st.write(f"Nombre total d'observations : {len(df)}")
        st.write("Statistiques descriptives :")
        st.dataframe(df.describe())
        
        # Ajout d'un histogramme de la distribution des classes
        st.subheader("Distribution des Classes")
        fig = px.histogram(df, x='target', 
                          title="Distribution des Classes dans la Base de Données",
                          labels={'target': 'Classe', 'count': 'Nombre d\'observations'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Interprétation de la distribution des classes
        st.subheader("Interprétation de la Distribution des Classes")
        st.markdown("""
La distribution des classes est équilibrée, ce qui est favorable pour l'apprentissage automatique et l'évaluation des modèles.
""")
    
    elif viz_type == "Matrice de Corrélation":
        st.header("Matrice de Corrélation")
        corr = df.corr()
        fig = px.imshow(corr, 
                       title="Matrice de Corrélation entre les Variables",
                       color_continuous_scale='RdBu',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
        
        # Interprétation spécifique de la matrice de corrélation
        st.subheader("Interprétation spécifique de la matrice de corrélation")
        st.markdown("""
- La matrice montre que la plupart des variables sont peu corrélées entre elles (hors diagonale), ce qui est positif pour l'apprentissage automatique.
- Aucune paire de variables ne présente de corrélation très forte (proche de 1 ou -1), donc chaque variable apporte une information spécifique.
- Par exemple, l'âge n'est pas fortement corrélé avec les autres variables, ce qui suggère une influence indépendante.
- La variable cible (`target`) n'est pas fortement corrélée à une seule variable, ce qui montre que le risque est multifactoriel.
""")
    
    elif viz_type == "Distribution des Variables":
        st.header("Distribution des Variables")
        
        col1, col2 = st.columns(2)
        with col1:
            variable = st.selectbox(
                "Sélectionnez une variable",
                df.select_dtypes(include=['float64', 'int64']).columns
            )
        with col2:
            plot_type = st.selectbox(
                "Type de graphique",
                ["Histogramme", "Box Plot", "Violin Plot"]
            )
        
        if plot_type == "Histogramme":
            fig = px.histogram(df, x=variable, color='target',
                             title=f"Distribution de {variable} par Classe",
                             marginal="box",
                             nbins=50)
        elif plot_type == "Box Plot":
            fig = px.box(df, x='target', y=variable,
                        title=f"Box Plot de {variable} par Classe")
        else:
            fig = px.violin(df, x='target', y=variable,
                          title=f"Violin Plot de {variable} par Classe")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Ajout des statistiques descriptives pour la variable sélectionnée
        st.subheader(f"Statistiques pour {variable}")
        st.dataframe(df.groupby('target')[variable].describe())
        
        # Interprétation spécifique de la variable sélectionnée
        st.subheader("Interprétation spécifique de la variable sélectionnée")
        if variable == "sex":
            st.markdown("""
- La majorité des individus sont des hommes (1), mais la maladie touche les deux sexes.
- La proportion de cas de maladie semble similaire dans les deux groupes, mais il faudrait regarder les pourcentages pour conclure.
""")
        elif variable == "age":
            st.markdown("""
- La distribution de l'âge est similaire pour les deux classes.
- Il n'y a pas de tranche d'âge où la maladie est nettement plus fréquente.
""")
        else:
            st.markdown("""
- Cette variable ne présente pas de différence flagrante entre les classes, mais peut contribuer à la prédiction en combinaison avec d'autres variables.
""")
    
    elif viz_type == "Analyse Bivariée":
        st.header("Analyse Bivariée")
        
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox(
                "Variable X",
                df.select_dtypes(include=['float64', 'int64']).columns
            )
        with col2:
            y_var = st.selectbox(
                "Variable Y",
                df.select_dtypes(include=['float64', 'int64']).columns
            )
        
        fig = px.scatter(df, x=x_var, y=y_var, color='target',
                        title=f"Relation entre {x_var} et {y_var}",
                        opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
        
        # Calcul et affichage de la corrélation
        correlation = df[x_var].corr(df[y_var])
        st.write(f"Corrélation entre {x_var} et {y_var} : {correlation:.3f}")
        
        # Interprétation spécifique de l'analyse bivariée
        st.subheader("Interprétation spécifique de l'analyse bivariée")
        st.markdown(f"""
- Le nuage de points montre une grande dispersion, sans tendance linéaire claire.
- Les deux classes sont mélangées, ce qui indique que la combinaison de {x_var} et {y_var} ne suffit pas à séparer les classes.
""")
    
    else:  # Analyse PCA
        st.header("Analyse en Composantes Principales")
        
        try:
            # Chargement des résultats PCA
            pca_results = pd.read_csv('pca_results.csv')
            
            # Visualisation de la variance expliquée
            fig = px.line(x=range(1, len(pca_results) + 1),
                         y=pca_results['Variance Expliquée'],
                         title="Variance Expliquée par les Composantes Principales")
            fig.update_layout(xaxis_title="Nombre de Composantes",
                            yaxis_title="Variance Expliquée")
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualisation des deux premières composantes
            fig = px.scatter(pca_results, x='PC1', y='PC2', color='target',
                            title="Projection sur les Deux Premières Composantes Principales",
                            opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
            
            # Interprétation de l'analyse PCA
            st.subheader("Interprétation de l'Analyse PCA")
            st.markdown("""
            ### Analyse des Composantes Principales :
            
            1. **Variance Expliquée**
               - La courbe montre la quantité d'information conservée
               - Le coude (elbow) indique le nombre optimal de composantes
               - La variance cumulée montre la perte d'information
            
            2. **Projection 2D**
               - Les clusters visibles indiquent des groupes naturels
               - La séparation des classes montre la qualité de la réduction
               - Les outliers peuvent être identifiés
            
            3. **Points d'Attention**
               - La perte d'information doit être évaluée
               - L'interprétabilité des composantes est importante
               - La qualité de la séparation des classes doit être considérée
            """)
            
        except FileNotFoundError:
            st.warning("Veuillez d'abord exécuter l'analyse PCA pour voir les résultats.")

except FileNotFoundError:
    st.error("Veuillez d'abord exécuter le script de prétraitement pour voir les visualisations.") 