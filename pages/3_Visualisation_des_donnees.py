import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Exploration des données", page_icon="")

st.title("Exploration et analyse des données")

try:
    # Chargement des données
    df = pd.read_csv('data/data.csv')

    # Affichage des informations générales
    st.sidebar.header("Informations sur le jeu de données")
    st.sidebar.write(f"Nombre total d'observations : {len(df)}")
    st.sidebar.write(f"Nombre de variables : {len(df.columns)}")

    # Sélection du type de visualisation
    viz_type = st.selectbox(
        "Type de visualisation",
        ["Aperçu des données", "Matrice de corrélation",
         "Répartition des variables", "Analyse bivariée",
         "Analyse en composantes principales (ACP)"]
    )

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
        "target": "Maladie cardiovasculaire",
    }
    df_plot = df.assign(
        Statut=df["target"].map({0: "Sans maladie", 1: "Maladie cardiovasculaire"})
    )

    if viz_type == "Aperçu des données":
        st.header("Aperçu des données")
        st.write("Aperçu des 5 premières lignes :")
        st.dataframe(df.head().rename(columns=feature_labels))
        st.write(f"Nombre total d'observations : {len(df)}")
        st.write("Statistiques descriptives :")
        descriptive_stats = df.describe().rename(columns=feature_labels)
        descriptive_stats.index = ["Effectif", "Moyenne", "Écart-type", "Minimum", "1er quartile", "Médiane", "3e quartile", "Maximum"]
        st.dataframe(descriptive_stats)

        # Ajout d'un histogramme de la distribution des classes
        st.subheader("Répartition des classes")
        fig = px.histogram(df_plot, x='Statut',
                          title="Répartition des classes dans le jeu de données",
                          labels={'Statut': 'Statut clinique', 'count': 'Nombre d’observations'})
        st.plotly_chart(fig, use_container_width=True)

        # Interprétation de la distribution des classes
        st.subheader("Interprétation de la répartition des classes")
        st.markdown("""
La distribution des classes est équilibrée, ce qui est favorable pour l'apprentissage automatique et l'évaluation des modèles.
""")

    elif viz_type == "Matrice de corrélation":
        st.header("Matrice de corrélation")
        corr = df.corr().rename(index=feature_labels, columns=feature_labels)
        fig = px.imshow(corr,
                       title="Matrice de corrélation entre les variables",
                       color_continuous_scale='RdBu',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)

        # Interprétation spécifique de la matrice de corrélation
        st.subheader("Interprétation de la matrice de corrélation")
        st.markdown("""
- La matrice montre que la plupart des variables sont peu corrélées entre elles (hors diagonale), ce qui est positif pour l'apprentissage automatique.
- Aucune paire de variables ne présente de corrélation très forte (proche de 1 ou -1), donc chaque variable apporte une information spécifique.
- Par exemple, l'âge n'est pas fortement corrélé avec les autres variables, ce qui suggère une influence indépendante.
- La variable cible (`target`) n'est pas fortement corrélée à une seule variable, ce qui montre que le risque est multifactoriel.
""")

    elif viz_type == "Répartition des variables":
        st.header("Répartition des variables")

        col1, col2 = st.columns(2)
        with col1:
            variable = st.selectbox(
                "Sélectionnez une variable",
                df.select_dtypes(include=['float64', 'int64']).columns,
                format_func=lambda col: feature_labels.get(col, col),
            )
        with col2:
            plot_type = st.selectbox(
                "Type de graphique",
                ["Histogramme", "Boîte à moustaches", "Diagramme en violon"]
            )

        if plot_type == "Histogramme":
            fig = px.histogram(df_plot, x=variable, color='Statut',
                             title=f"Répartition de {feature_labels.get(variable, variable)} par classe",
                             labels={"Statut": "Statut clinique", variable: feature_labels.get(variable, variable)},
                             marginal="box",
                             nbins=50)
        elif plot_type == "Boîte à moustaches":
            fig = px.box(df_plot, x='Statut', y=variable,
                        title=f"Boîte à moustaches de {feature_labels.get(variable, variable)} par classe",
                        labels={"Statut": "Statut clinique", variable: feature_labels.get(variable, variable)})
        else:
            fig = px.violin(df_plot, x='Statut', y=variable,
                          title=f"Diagramme en violon de {feature_labels.get(variable, variable)} par classe",
                          labels={"Statut": "Statut clinique", variable: feature_labels.get(variable, variable)})

        st.plotly_chart(fig, use_container_width=True)

        # Ajout des statistiques descriptives pour la variable sélectionnée
        st.subheader(f"Statistiques pour {feature_labels.get(variable, variable)}")
        variable_stats = df.groupby('target')[variable].describe()
        variable_stats.index = ["Sans maladie", "Avec maladie"]
        variable_stats = variable_stats.rename(columns={
            "count": "Effectif", "mean": "Moyenne", "std": "Écart-type",
            "min": "Minimum", "25%": "1er quartile", "50%": "Médiane",
            "75%": "3e quartile", "max": "Maximum",
        })
        st.dataframe(variable_stats)

        # Interprétation spécifique de la variable sélectionnée
        st.subheader("Interprétation de la variable sélectionnée")
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

    elif viz_type == "Analyse bivariée":
        st.header("Analyse bivariée")

        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox(
                "Variable X",
                df.select_dtypes(include=['float64', 'int64']).columns,
                format_func=lambda col: feature_labels.get(col, col),
            )
        with col2:
            y_var = st.selectbox(
                "Variable Y",
                df.select_dtypes(include=['float64', 'int64']).columns,
                format_func=lambda col: feature_labels.get(col, col),
            )

        fig = px.scatter(df_plot, x=x_var, y=y_var, color='Statut',
                        title=f"Relation entre {feature_labels.get(x_var, x_var)} et {feature_labels.get(y_var, y_var)}",
                        labels={"Statut": "Statut clinique", x_var: feature_labels.get(x_var, x_var), y_var: feature_labels.get(y_var, y_var)},
                        opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

        # Calcul et affichage de la corrélation
        correlation = df[x_var].corr(df[y_var])
        st.write(f"Corrélation entre {feature_labels.get(x_var, x_var)} et {feature_labels.get(y_var, y_var)} : {correlation:.3f}")

        # Interprétation spécifique de l'analyse bivariée
        st.subheader("Interprétation de l’analyse bivariée")
        st.markdown(f"""
- Le nuage de points montre une grande dispersion, sans tendance linéaire claire.
- Les deux classes sont mélangées, ce qui indique que la combinaison de {x_var} et {y_var} ne suffit pas à séparer les classes.
""")

    else:  # Analyse en composantes principales
        st.header("Analyse en composantes principales (ACP)")

        try:
            # Chargement des résultats PCA
            pca_results = pd.read_csv('pca_results.csv')

            # Visualisation de la variance expliquée
            fig = px.line(x=range(1, len(pca_results) + 1),
                         y=pca_results['Variance Expliquée'],
                         title="Variance expliquée par les composantes principales")
            fig.update_layout(xaxis_title="Nombre de composantes",
                            yaxis_title="Variance expliquée")
            st.plotly_chart(fig, use_container_width=True)

            # Visualisation des deux premières composantes
            pca_results["Statut"] = pca_results["target"].map(
                {0: "Sans maladie", 1: "Maladie cardiovasculaire"}
            )
            fig = px.scatter(pca_results, x='PC1', y='PC2', color='Statut',
                            title="Projection sur les deux premières composantes principales",
                            labels={"PC1": "Composante principale 1", "PC2": "Composante principale 2", "Statut": "Statut clinique"},
                            opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)

            # Interprétation de l'analyse PCA
            st.subheader("Interprétation de l’analyse ACP")
            st.markdown("""
            ### Analyse des composantes principales :

            1. **Variance Expliquée**
               - La courbe montre la quantité d'information conservée
               - Le point d’inflexion indique le nombre optimal de composantes
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
            st.warning("Veuillez d’abord exécuter l’analyse ACP pour afficher les résultats.")

except FileNotFoundError:
    st.error("Veuillez d’abord exécuter le prétraitement pour afficher les visualisations.")
