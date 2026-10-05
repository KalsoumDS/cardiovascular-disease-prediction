import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import preprocess_data
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Prédiction des Maladies Cardiovasculaires",
    page_icon="",
    layout="wide"
)


def compute_shap_contributions(model, processed_df):
    """Calcule les contributions locales SHAP ou tree feature contributions pour le patient."""
    feature_labels = {
        'age': 'Âge',
        'sex': 'Sexe',
        'resting bp s': 'Pression artérielle',
        'cholesterol': 'Cholestérol',
        'fasting blood sugar': 'Glycémie à jeun',
        'max heart rate': 'Fréquence cardiaque max',
        'exercise angina': 'Angine à l effort',
        'oldpeak': 'Dépression ST (oldpeak)',
        'chest pain type_1': 'Douleur : Typique',
        'chest pain type_2': 'Douleur : Atypique',
        'chest pain type_3': 'Douleur : Non-angineuse',
        'chest pain type_4': 'Douleur : Asymptomatique',
        'resting ecg_0': 'ECG : Normal',
        'resting ecg_1': 'ECG : Anomalie ST-T',
        'resting ecg_2': 'ECG : Hypertrophie VG',
        'ST slope_1': 'Pente ST : Ascendante',
        'ST slope_2': 'Pente ST : Plate',
        'ST slope_3': 'Pente ST : Descendante'
    }
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(processed_df)
        if isinstance(shap_vals, list) and len(shap_vals) > 1:
            vals = shap_vals[1][0]
        elif hasattr(shap_vals, "values"):
            vals = shap_vals.values[0, :, 1] if len(shap_vals.shape) == 3 else shap_vals.values[0]
        else:
            vals = shap_vals[0]
    except Exception:
        importances = getattr(model, 'feature_importances_', np.ones(processed_df.shape[1]) / processed_df.shape[1])
        vals = (processed_df.values[0] - 0.0) * importances
        
    contribs = []
    for col, v in zip(processed_df.columns, vals):
        label = feature_labels.get(col, col)
        contribs.append({'feature': label, 'value': float(v)})
    
    df_contribs = pd.DataFrame(contribs)
    df_contribs['abs_val'] = df_contribs['value'].abs()
    df_contribs = df_contribs.sort_values(by='abs_val', ascending=True).tail(8)
    return df_contribs


def plot_shap_waterfall(df_contribs):
    """Génère un graphique horizontal de contribution SHAP."""
    colors = ['#ef4444' if v > 0 else '#10b981' for v in df_contribs['value']]
    fig = go.Figure(go.Bar(
        x=df_contribs['value'],
        y=df_contribs['feature'],
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.2)', width=1)),
        text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in df_contribs['value']],
        textposition='outside'
    ))
    fig.update_layout(
        title="<b>Explicabilité Clinique SHAP — Impact individuel sur la prédiction</b>",
        xaxis_title="Contribution au risque (+ augmente le risque | - facteur protecteur)",
        yaxis_title="Facteur de risque clinique",
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        height=380,
        margin=dict(l=20, r=40, t=50, b=40)
    )
    return fig

# CSS personnalisé
st.markdown("""
    <style>
    /* Styles généraux */
    .stApp {
        background-color: #1a1a1a;
    }
    
    /* Container du titre */
    .main-title-container {
        background: linear-gradient(135deg, #3498db, #2980b9);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem auto;
        max-width: 800px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* Titre principal */
    .main-title {
        color: white !important;
        font-size: 2rem !important;
        font-weight: bold !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }

    /* Container des cartes */
    .cards-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 2rem;
        margin: 2rem auto;
        max-width: 1200px;
        padding: 0 2rem;
    }

    /* Cartes de service */
    .service-card {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
    }

    .service-icon {
        font-size: 2.5rem;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .service-card h3 {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 500;
        margin: 0;
        line-height: 1.2;
    }

    .service-card p {
        color: #cccccc;
        font-size: 1rem;
        margin: 0;
        line-height: 1.5;
        max-width: 250px;
    }

    /* Section Comment ça marche */
    .steps-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .steps-container h2 {
        color: #ffffff;
    }

    .step-number {
        background: #3498db;
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem auto;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    /* Bouton d'action */
    .action-button {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        padding: 1rem 2rem;
        border-radius: 30px;
        border: none;
        font-weight: bold;
        cursor: pointer;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        width: 100%;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.3);
    }

    /* Note d'avertissement */
    .warning-note {
        background: rgba(255, 193, 7, 0.1);
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 2rem 0;
        color: #ffc107;
    }

    .warning-note h4 {
        color: #ffc107;
        margin-bottom: 0.5rem;
    }

    .warning-note p {
        color: #cccccc;
        margin: 0;
    }

    /* Animations */
    @keyframes slideDown {
        from {
            transform: translateY(-20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Prédiction", "Visualisation des Données", "À Propos"]
)

if page == "Accueil":
    # Titre dans un container bleu
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #3498db, #2980b9); padding: 15px; border-radius: 10px; text-align: center;'>
                <h1 style='color: white; margin: 0; font-size: 1.8rem;'>Système de Prédiction des Maladies Cardiovasculaires</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Espacement
    st.write("")
    st.write("")

    # Trois colonnes pour les services
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div style='text-align: center;'>
                <h2 style='font-size: 2.5rem; margin-bottom: 0.5rem;'></h2>
                <h3 style='margin-top: 0;'>Évaluation Rapide</h3>
                <p style='color: #cccccc;'>Obtenez une évaluation personnalisée de votre risque cardiovasculaire</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='text-align: center;'>
                <h2 style='font-size: 2.5rem; margin-bottom: 0.5rem;'></h2>
                <h3 style='margin-top: 0;'>Analyse Détaillée</h3>
                <p style='color: #cccccc;'>Visualisez et comprenez vos indicateurs de santé</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style='text-align: center;'>
                <h2 style='font-size: 2.5rem; margin-bottom: 0.5rem;'></h2>
                <h3 style='margin-top: 0;'>Conseils Personnalisés</h3>
                <p style='color: #cccccc;'>Recevez des recommandations adaptées à votre profil</p>
            </div>
        """, unsafe_allow_html=True)

    # Espacement
    st.write("")
    st.write("")

    # Section Comment ça marche
    st.header("Comment ça marche ?")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div style='text-align: center;'>
                <h2 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>1⃣</h2>
                <h3 style='margin-top: 0;'>Entrez vos données</h3>
                <p style='color: #cccccc;'>Remplissez un formulaire simple avec vos informations médicales de base</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='text-align: center;'>
                <h2 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>2⃣</h2>
                <h3 style='margin-top: 0;'>Analyse instantanée</h3>
                <p style='color: #cccccc;'>Notre système analyse vos données et calcule votre niveau de risque</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style='text-align: center;'>
                <h2 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>3⃣</h2>
                <h3 style='margin-top: 0;'>Résultats et conseils</h3>
                <p style='color: #cccccc;'>Obtenez vos résultats et des recommandations pour améliorer votre santé</p>
            </div>
        """, unsafe_allow_html=True)

    # Bouton d'action
    st.write("")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button(" Commencer l'évaluation", use_container_width=True):
            st.session_state.page = "Prédiction"

    # Note importante
    st.warning("""
         **Note importante**
        
        Cette application est un outil d'aide à la décision. Elle ne remplace pas l'avis d'un professionnel de santé.
        Consultez toujours votre médecin pour un diagnostic complet.
    """)

elif page == "Prédiction":
    st.markdown("""
        <div style='text-align: center; padding: 1rem; margin-bottom: 2rem;'>
            <h1 style='color: #ffffff;'>Prédiction du Risque Cardiovasculaire</h1>
            <p style='color: #ffffff; font-size: 1.1rem;'>
                Remplissez le formulaire ci-dessous pour obtenir une évaluation personnalisée
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Conteneur principal avec style
    st.markdown("""
        <div style='background-color: #2c2c2c; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
    """, unsafe_allow_html=True)
    
    # Formulaire de saisie
    col1, col2 = st.columns(2)
    
    # Dictionnaire de conversion pour les types de douleur thoracique
    chest_pain_mapping = {
        "Typique": 1,
        "Atypique": 2,
        "Non-angineux": 3,
        "Asymptomatique": 4
    }
    
    # Dictionnaire de conversion pour les résultats ECG
    ecg_mapping = {
        "Normal": 0,
        "Anomalie ST-T": 1,
        "Hypertrophie ventriculaire gauche": 2
    }
    
    # Dictionnaire de conversion pour la pente ST
    st_slope_mapping = {
        "Ascendante": 1,
        "Plate": 2,
        "Descendante": 3
    }
    
    with col1:
        st.markdown("<h3 style='color: #3498db; margin-bottom: 1rem;'>Informations Personnelles</h3>", unsafe_allow_html=True)
        age = st.number_input("Âge", min_value=18, max_value=100, value=30)
        sex = st.selectbox("Sexe", ["Homme", "Femme"])
        chest_pain = st.selectbox("Type de douleur thoracique", 
                                list(chest_pain_mapping.keys()))
        resting_bp = st.number_input("Pression artérielle au repos (mmHg)", 
                                   min_value=90, max_value=200, value=120)
        oldpeak = st.number_input("Dépression ST (oldpeak)", 
                                min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    
    with col2:
        st.markdown("<h3 style='color: #3498db; margin-bottom: 1rem;'>Paramètres Médicaux</h3>", unsafe_allow_html=True)
        cholesterol = st.number_input("Cholestérol (mg/dl)", 
                                    min_value=100, max_value=600, value=200)
        fasting_bs = st.selectbox("Glycémie à jeun", ["< 120 mg/dl", "> 120 mg/dl"])
        resting_ecg = st.selectbox("Résultats ECG au repos", 
                                 list(ecg_mapping.keys()))
        max_hr = st.number_input("Fréquence cardiaque maximale", 
                               min_value=60, max_value=220, value=150)
        st_slope = st.selectbox("Pente du segment ST", 
                              list(st_slope_mapping.keys()))
        exercise_angina = st.selectbox("Angine induite par l'exercice", 
                                     ["Non", "Oui"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Espace pour le bouton
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bouton de prédiction avec style
    if st.button("Obtenir la Prédiction", key="prediction_button"):
        with st.spinner("Analyse en cours..."):
            try:
                # Chargement du modèle
                model = joblib.load('models/random_forest_model.joblib')
                
                # Préparation des données
                input_data = pd.DataFrame({
                    'age': [age],
                    'sex': [1 if sex == "Homme" else 0],
                    'chest pain type': [chest_pain_mapping[chest_pain]],
                    'resting bp s': [resting_bp],
                    'cholesterol': [cholesterol],
                    'fasting blood sugar': [1 if fasting_bs == "> 120 mg/dl" else 0],
                    'resting ecg': [ecg_mapping[resting_ecg]],
                    'max heart rate': [max_hr],
                    'exercise angina': [1 if exercise_angina == "Oui" else 0],
                    'oldpeak': [oldpeak],
                    'ST slope': [st_slope_mapping[st_slope]]
                })
                
                processed_data = preprocess_data(input_data, is_training=False)
                prediction = model.predict(processed_data)[0]
                probability = model.predict_proba(processed_data)[0][1]
            except Exception as e:
                st.error(f"Une erreur est survenue : {str(e)}")
            
            # Affichage des résultats avec style
            st.markdown("""
                <div style='background-color: #2c2c2c; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); margin-top: 2rem;'>
                    <h2 style='color: #ffffff; text-align: center; margin-bottom: 1.5rem;'>Résultats de la Prédiction</h2>
            """, unsafe_allow_html=True)
            
            # Affichage du résultat avec style
            if prediction == 1:
                st.markdown(f"""
                    <div style='background-color: #2c1c1c; padding: 1rem; border-radius: 5px; text-align: center; margin-bottom: 1rem; border-left: 5px solid #dc3545;'>
                        <h3 style='color: #dc3545; margin: 0;'> Risque élevé de maladie cardiovasculaire</h3>
                        <p style='color: #ffffff; margin: 0.5rem 0 0 0;'>Probabilité: {probability*100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background-color: #1c2c1c; padding: 1rem; border-radius: 5px; text-align: center; margin-bottom: 1rem; border-left: 5px solid #28a745;'>
                        <h3 style='color: #28a745; margin: 0;'> Risque faible de maladie cardiovasculaire</h3>
                        <p style='color: #ffffff; margin: 0.5rem 0 0 0;'>Probabilité: {probability*100:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
            

            # --- Module Explicabilité SHAP Clinique ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #3498db; margin-top: 1rem;'>Explicabilité Clinique des Prédictions (SHAP TreeExplainer)</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #cbd5e1;'>Décomposition transparente des facteurs physiologiques qui influencent la probabilité de risque pour ce patient spécifique.</p>", unsafe_allow_html=True)
            
            try:
                df_contribs = compute_shap_contributions(model, processed_data)
                fig_shap = plot_shap_waterfall(df_contribs)
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as ex_shap:
                st.info(f"Analyse SHAP : {ex_shap}")

            # --- Module Simulateur Clinique What-If ---
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🩺 Simulateur de Prévention Clinique « What-If » (Interventions Ciblées)", expanded=True):
                st.markdown("<p style='color: #cbd5e1;'>Simulez l'impact prévisionnel d'interventions thérapeutiques ou de modifications de mode de vie sur ce patient :</p>", unsafe_allow_html=True)
                w_col1, w_col2, w_col3 = st.columns(3)
                with w_col1:
                    new_bp = st.slider("Cible Pression Artérielle (mmHg)", min_value=90, max_value=200, value=min(int(resting_bp), 120), step=5)
                with w_col2:
                    new_chol = st.slider("Cible Cholestérol (mg/dl)", min_value=100, max_value=400, value=min(int(cholesterol), 180), step=10)
                with w_col3:
                    new_angina = st.selectbox("Arrêt Angine d'effort", ["Non", "Oui"], index=0 if exercise_angina == "Non" else 1)
                
                sim_input = input_data.copy()
                sim_input['resting bp s'] = [new_bp]
                sim_input['cholesterol'] = [new_chol]
                sim_input['exercise angina'] = [1 if new_angina == "Oui" else 0]
                
                sim_processed = preprocess_data(sim_input, is_training=False)
                new_prob = model.predict_proba(sim_processed)[0][1]
                delta_prob = (new_prob - probability) * 100
                
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
                    st.metric(
                        label="Nouveau Risque Cardiovasculaire",
                        value=f"{new_prob*100:.1f}%",
                        delta=f"{delta_prob:.1f}%",
                        delta_color="inverse"
                    )
                with res_col2:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=new_prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        delta={'reference': probability * 100, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#ef4444" if new_prob >= 0.5 else "#10b981"},
                            'steps': [
                                {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                                {'range': [30, 60], 'color': "rgba(245, 158, 11, 0.2)"},
                                {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                            ],
                            'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.75, 'value': 50}
                        }
                    ))
                    fig_gauge.update_layout(template="plotly_dark", height=200, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)

            # Recommandations avec style
            st.markdown("<h3 style='color: #ffffff; margin-top: 1.5rem;'>Recommandations Personnalisées</h3>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown("""
                    <div style='background-color: #1a1a1a; padding: 1rem; border-radius: 5px; margin-top: 1rem;'>
                        <ul style='color: #ffffff; margin: 0; padding-left: 1.5rem;'>
                            <li style='margin-bottom: 0.5rem;'> Consultez un médecin pour un examen approfondi</li>
                            <li style='margin-bottom: 0.5rem;'> Adoptez un régime alimentaire sain et équilibré</li>
                            <li style='margin-bottom: 0.5rem;'> Pratiquez une activité physique régulière adaptée</li>
                            <li style='margin-bottom: 0.5rem;'> Évitez le tabac et limitez la consommation d'alcool</li>
                            <li style='margin-bottom: 0.5rem;'> Apprenez à gérer votre stress au quotidien</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='background-color: #1a1a1a; padding: 1rem; border-radius: 5px; margin-top: 1rem;'>
                        <ul style='color: #ffffff; margin: 0; padding-left: 1.5rem;'>
                            <li style='margin-bottom: 0.5rem;'> Continuez à maintenir vos bonnes habitudes de vie</li>
                            <li style='margin-bottom: 0.5rem;'> Surveillez régulièrement vos paramètres de santé</li>
                            <li style='margin-bottom: 0.5rem;'> Effectuez des bilans de santé réguliers</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "Visualisation des Données":
    st.header("Visualisation des Données")
    
    # Chargement des données
    try:
        df = pd.read_csv('data/data.csv')
        
        # Sélection du type de visualisation
        viz_type = st.selectbox(
            "Type de Visualisation",
            ["Matrice de Corrélation", "Distribution des Variables", "Importance des Caractéristiques"]
        )
        
        if viz_type == "Matrice de Corrélation":
            st.subheader("Matrice de Corrélation")
            fig = px.imshow(df.corr(), 
                          title="Matrice de Corrélation entre les Variables",
                          color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Distribution des Variables":
            st.subheader("Distribution des Variables")
            selected_var = st.selectbox("Sélectionnez une Variable", 
                                      df.select_dtypes(include=['float64', 'int64']).columns)
            fig = px.histogram(df, x=selected_var, color='target',
                             title=f"Distribution de {selected_var} par Classe")
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.subheader("Importance des Caractéristiques")
            try:
                model = joblib.load('models/random_forest_model.joblib')
                feature_importance = pd.DataFrame({
                    'Feature': df.columns[:-1],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(feature_importance, 
                           x='Feature', 
                           y='Importance',
                           title="Importance des Caractéristiques dans la Prédiction")
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning("Veuillez d'abord entraîner le modèle pour voir l'importance des caractéristiques.")
    
    except FileNotFoundError:
        st.error("Fichier de données non trouvé. Veuillez d'abord exécuter le script de prétraitement.")

else:  # À Propos
    st.header("À Propos")
    st.markdown("""
    ### Objectif du Projet
    Ce projet vise à développer un système de prédiction des risques de maladies cardiovasculaires en utilisant des techniques de Machine Learning.
    
    ### Technologies Utilisées
    - Python
    - Scikit-learn
    - Streamlit
    - Pandas
    - NumPy
    - Matplotlib/Seaborn
    - Plotly
    
    ### Modèles Implémentés
    - Régression Logistique
    - KNN
    - Arbre de Décision
    - Random Forest
    - KMeans
    - PCA
    
    ### Avertissement
    Cette application est destinée à des fins éducatives et ne remplace pas un diagnostic médical professionnel.
    Consultez toujours un médecin pour des conseils médicaux appropriés.
    """) 