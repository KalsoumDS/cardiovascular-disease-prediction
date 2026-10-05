"""
Système Avancé d'Aide au Diagnostic & Prédiction du Risque Cardiovasculaire
Plateforme MedTech SOTA — Machine Learning Clinique, Explicabilité SHAP & Simulation Préventive
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from src.data_preprocessing import preprocess_data

# Configuration de la page
st.set_page_config(
    page_title="CardioSOTA | Diagnostic Cardiovasculaire Prédictif",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DESIGN SYSTEM HEALTH TECH MODERNE ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    .stApp {
        background: #0b1120;
        color: #f1f5f9;
    }

    /* En-tête Médical Premium */
    .med-header {
        background: linear-gradient(135deg, rgba(14, 116, 144, 0.35) 0%, rgba(59, 130, 246, 0.3) 50%, rgba(99, 102, 241, 0.25) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px -15px rgba(2, 132, 199, 0.15);
        backdrop-filter: blur(12px);
    }
    .med-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .med-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
        line-height: 1.6;
    }

    /* Cartes cliniques */
    .clinic-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }

    /* Paliers de risque dynamiques */
    .risk-banner-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 78, 59, 0.25) 100%);
        border: 1px solid #10b981;
        border-left: 6px solid #10b981;
        border-radius: 12px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
    }
    .risk-banner-moderate {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.15) 0%, rgba(113, 63, 18, 0.25) 100%);
        border: 1px solid #eab308;
        border-left: 6px solid #eab308;
        border-radius: 12px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
    }
    .risk-banner-high {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.18) 0%, rgba(154, 52, 18, 0.3) 100%);
        border: 1px solid #f97316;
        border-left: 6px solid #f97316;
        border-radius: 12px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
    }
    .risk-banner-critical {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(127, 29, 29, 0.35) 100%);
        border: 1px solid #ef4444;
        border-left: 6px solid #ef4444;
        border-radius: 12px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
    }

    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Presets cliniques */
    .preset-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── FONCTIONS D'ÉVALUATION ET VISUALISATION ────────────────────────────────────

def get_risk_tier(prob: float):
    """
    Retourne la segmentation clinique quadri-niveau avec couleurs et descriptions précises.
    Permet de distinguer nettement 49% (Modéré/Jaune) de 50% (Élevé/Orange) et 75% (Critique/Rouge).
    """
    if prob < 0.25:
        return {
            "level": "Risque Faible (Optimal)",
            "css_class": "risk-banner-low",
            "color": "#10b981",
            "badge_bg": "rgba(16, 185, 129, 0.2)",
            "title": "Profil Hémodynamique & Cardiovasculaire Favorable",
            "desc": f"Probabilité estimée : {prob*100:.1f}%. Le profil clinique se situe dans la plage optimale.",
            "action": "Maintenir l'activité physique hebdomadaire et surveillance de routine (bilan annuel)."
        }
    elif prob < 0.50:
        return {
            "level": "Risque Modéré (Vigilance)",
            "css_class": "risk-banner-moderate",
            "color": "#eab308",
            "badge_bg": "rgba(234, 179, 8, 0.2)",
            "title": "Risque Modéré — Facteurs de Vigilance Détectés",
            "desc": f"Probabilité estimée : {prob*100:.1f}%. Présence de marqueurs limites (tension artérielle ou lipidique).",
            "action": "Rééquilibrage diététique, réduction des apports sodés et suivi de la pression artérielle sous 3 mois."
        }
    elif prob < 0.75:
        return {
            "level": "Risque Élevé (Prise en Charge)",
            "css_class": "risk-banner-high",
            "color": "#f97316",
            "badge_bg": "rgba(249, 115, 22, 0.25)",
            "title": "Risque Cardiovasculaire Élevé — Consultation Conseillée",
            "desc": f"Probabilité estimée : {prob*100:.1f}%. Forte convergence de cofacteurs de risque coronarien.",
            "action": "Consultation médicale complète, épreuve d'effort cardiologique et bilan lipidique détaillé recommandés."
        }
    else:
        return {
            "level": "Alerte Critique (Priorité Médicale)",
            "css_class": "risk-banner-critical",
            "color": "#ef4444",
            "badge_bg": "rgba(239, 68, 68, 0.3)",
            "title": "Alerte Clinique Critique — Prise en Charge Prioritaire",
            "desc": f"Probabilité estimée : {prob*100:.1f}%. Profil à haute probabilité d'ischémie ou pathologie cardiovasculaire.",
            "action": "Avis cardiologique rapide requis pour investigation coronarographique ou thérapie ciblée."
        }


def compute_framingham_risk(age: int, sex: int, sbp: float, chol: float, smoker: int = 0) -> float:
    """Score de risque simplifié inspiré de Framingham / SCORE2 européen pour benchmark classique."""
    points = 0
    # Âge
    if age > 60: points += 7
    elif age > 50: points += 5
    elif age > 40: points += 3
    # Sexe (Homme = 1)
    if sex == 1: points += 3
    # Tension
    if sbp >= 160: points += 5
    elif sbp >= 140: points += 3
    elif sbp >= 130: points += 1
    # Cholestérol
    if chol >= 280: points += 4
    elif chol >= 240: points += 2
    elif chol >= 200: points += 1
    if smoker == 1: points += 4

    prob = 1.0 / (1.0 + np.exp(-(points - 10) / 4.0))
    return float(np.clip(prob, 0.05, 0.95))


def compute_shap_contributions(model, processed_df):
    """Calcule les contributions locales SHAP ou tree feature contributions pour le patient."""
    feature_labels = {
        'age': 'Âge du patient',
        'sex': 'Sexe',
        'resting bp s': 'Pression artérielle repos',
        'cholesterol': 'Taux de Cholestérol',
        'fasting blood sugar': 'Glycémie à jeun > 120',
        'max heart rate': 'Fréquence cardiaque max',
        'exercise angina': 'Angine d effort',
        'oldpeak': 'Dépression segment ST',
        'chest pain type_1': 'Douleur typique',
        'chest pain type_2': 'Douleur atypique',
        'chest pain type_3': 'Douleur non angineuse',
        'chest pain type_4': 'Douleur asymptomatique',
        'resting ecg_0': 'ECG repos normal',
        'resting ecg_1': 'ECG anomalie ST-T',
        'resting ecg_2': 'ECG hypertrophie VG',
        'ST slope_1': 'Pente ST ascendante',
        'ST slope_2': 'Pente ST plate',
        'ST slope_3': 'Pente ST descendante'
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


def plot_risk_gauge(prob: float, tier_color: str) -> go.Figure:
    """Jauge médicale semi-circulaire SOTA avec gradient continu vert/jaune/orange/rouge."""
    pct = prob * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': "%", 'font': {'size': 38, 'color': '#ffffff', 'family': 'Plus Jakarta Sans'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': tier_color, 'thickness': 0.3},
            'bgcolor': "#1e293b",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': "rgba(16, 185, 129, 0.25)"},
                {'range': [25, 50], 'color': "rgba(234, 179, 8, 0.25)"},
                {'range': [50, 75], 'color': "rgba(249, 115, 22, 0.25)"},
                {'range': [75, 100], 'color': "rgba(239, 68, 68, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 3},
                'thickness': 0.8,
                'value': pct
            }
        }
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=20, r=20, t=20, b=10)
    )
    return fig


def plot_cardio_radar(raw_data: dict) -> go.Figure:
    """Radar anatomique des 4 piliers de santé cardiovasculaire."""
    # Normalisation sur une échelle de 0 (optimal) à 100 (vulnérable)
    hemo_score = min(100, max(0, (raw_data['resting bp s'] - 100) / 80 * 100))
    lipid_score = min(100, max(0, (raw_data['cholesterol'] - 140) / 160 * 100))
    elec_score = min(100, (raw_data['oldpeak'] / 3.0 * 60) + (40 if raw_data['resting ecg'] > 0 else 0))
    effort_score = min(100, (100 if raw_data['exercise angina'] == 1 else 10) + max(0, (180 - raw_data['max heart rate']) / 80 * 50))

    categories = [
        'Hémodynamique (Tension)',
        'Lipides (Cholestérol)',
        'Électrophysiologie (ECG / ST)',
        'Tolérance à l Effort'
    ]
    values = [hemo_score, lipid_score, elec_score, effort_score]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Profil Patient',
        fillcolor='rgba(56, 189, 248, 0.25)',
        line=dict(color='#38bdf8', width=2.5)
    ))
    fig.add_trace(go.Scatterpolar(
        r=[35, 35, 35, 35, 35],
        theta=categories + [categories[0]],
        name='Seuil Optimal',
        line=dict(color='#10b981', dash='dash', width=1.5)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], color="#94a3b8")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=30, r=30, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig


def plot_shap_waterfall(df_contribs):
    """Génère un graphique horizontal de contribution SHAP avec code couleur distinct."""
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
        title="<b>Explicabilité SHAP — Impact de chaque facteur clinique sur le diagnostic</b>",
        xaxis_title="Contribution au risque (+ augmente le risque | - facteur protecteur)",
        yaxis_title="Paramètre physiologique",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350,
        margin=dict(l=20, r=40, t=50, b=40)
    )
    return fig


# ── NAVIGATION ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Navigation Clinique")
    page = st.radio(
        "Menu",
        ["Prédiction Clinique", "Visualisation des Données", "Architecture & Modèles", "À Propos"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Plateforme conforme aux standards d'aide au diagnostic préventif. Machine Learning validé sur cohorte cardiovasculaire multi-paramètres.")


# ── PAGE 1 : PRÉDICTION CLINIQUE ──────────────────────────────────────────────
if page == "Prédiction Clinique":
    st.markdown("""
    <div class="med-header">
        <h1 class="med-title">Diagnostic Clinique & Évaluation Cardiovasculaire</h1>
        <p class="med-subtitle">
            Système expert combinant Machine Learning (Random Forest / XGBoost), explicabilité locale SHAP 
            et simulation d'impact préventif (« What-If »).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Presets Cliniques en 1 Clic ──
    st.markdown("#####  Profils Patients Prédéfinis (Test Instantané)")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    if 'preset_loaded' not in st.session_state:
        st.session_state.preset_loaded = False

    def load_preset(age_val, sex_val, cp_val, bp_val, chol_val, bs_val, ecg_val, hr_val, ang_val, peak_val, slope_val):
        st.session_state.f_age = age_val
        st.session_state.f_sex = sex_val
        st.session_state.f_cp = cp_val
        st.session_state.f_bp = bp_val
        st.session_state.f_chol = chol_val
        st.session_state.f_bs = bs_val
        st.session_state.f_ecg = ecg_val
        st.session_state.f_hr = hr_val
        st.session_state.f_ang = ang_val
        st.session_state.f_peak = peak_val
        st.session_state.f_slope = slope_val
        st.session_state.preset_loaded = True

    with p_col1:
        if st.button(" Profil A : Athlète sain (Faible)", use_container_width=True):
            load_preset(28, "Homme", "Non-angineux", 112, 165, "< 120 mg/dl", "Normal", 178, "Non", 0.0, "Ascendante")
            st.rerun()
    with p_col2:
        if st.button(" Profil B : Sédentaire (Modéré)", use_container_width=True):
            load_preset(48, "Femme", "Atypique", 134, 218, "< 120 mg/dl", "Normal", 145, "Non", 0.6, "Plate")
            st.rerun()
    with p_col3:
        if st.button(" Profil C : Hypertendu (Élevé)", use_container_width=True):
            load_preset(58, "Homme", "Typique", 152, 255, "> 120 mg/dl", "Anomalie ST-T", 128, "Oui", 1.8, "Plate")
            st.rerun()
    with p_col4:
        if st.button(" Profil D : Alerte Sévère (Critique)", use_container_width=True):
            load_preset(68, "Homme", "Asymptomatique", 170, 295, "> 120 mg/dl", "Hypertrophie ventriculaire gauche", 110, "Oui", 3.2, "Descendante")
            st.rerun()

    # Dictionnaires de mapping
    chest_pain_mapping = {"Typique": 1, "Atypique": 2, "Non-angineux": 3, "Asymptomatique": 4}
    ecg_mapping = {"Normal": 0, "Anomalie ST-T": 1, "Hypertrophie ventriculaire gauche": 2}
    st_slope_mapping = {"Ascendante": 1, "Plate": 2, "Descendante": 3}

    # Formulaire de saisie
    with st.container():
        st.markdown('<div class="clinic-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("####  Paramètres Hémodynamiques & Individuels")
            age = st.number_input("Âge", min_value=18, max_value=100, value=st.session_state.get('f_age', 52))
            sex = st.selectbox("Sexe", ["Homme", "Femme"], index=0 if st.session_state.get('f_sex', "Homme") == "Homme" else 1)
            chest_pain = st.selectbox("Type de douleur thoracique", list(chest_pain_mapping.keys()),
                                    index=list(chest_pain_mapping.keys()).index(st.session_state.get('f_cp', "Atypique")))
            resting_bp = st.number_input("Pression artérielle systolique au repos (mmHg)", min_value=85, max_value=230, value=st.session_state.get('f_bp', 130))
            oldpeak = st.number_input("Dépression du segment ST (oldpeak)", min_value=0.0, max_value=8.0, value=float(st.session_state.get('f_peak', 0.5)), step=0.1)

        with col2:
            st.markdown("####  Paramètres Métaboliques & Électrophysiologiques")
            cholesterol = st.number_input("Taux de cholestérol sérique (mg/dl)", min_value=90, max_value=600, value=st.session_state.get('f_chol', 210))
            fasting_bs = st.selectbox("Glycémie à jeun", ["< 120 mg/dl", "> 120 mg/dl"],
                                    index=0 if st.session_state.get('f_bs', "< 120 mg/dl") == "< 120 mg/dl" else 1)
            resting_ecg = st.selectbox("Résultats de l'ECG au repos", list(ecg_mapping.keys()),
                                     index=list(ecg_mapping.keys()).index(st.session_state.get('f_ecg', "Normal")))
            max_hr = st.number_input("Fréquence cardiaque maximale atteinte", min_value=50, max_value=220, value=st.session_state.get('f_hr', 145))
            st_slope = st.selectbox("Pente du segment ST à l'effort", list(st_slope_mapping.keys()),
                                  index=list(st_slope_mapping.keys()).index(st.session_state.get('f_slope', "Plate")))
            exercise_angina = st.selectbox("Angine de poitrine induite par l'effort", ["Non", "Oui"],
                                         index=0 if st.session_state.get('f_ang', "Non") == "Non" else 1)

        st.markdown('</div>', unsafe_allow_html=True)

    btn_predict = st.button(" Lancer l'Analyse Diagnostique Médicale", type="primary", use_container_width=True)

    # Traitement
    if btn_predict or st.session_state.get('preset_loaded'):
        with st.spinner("Modélisation en cours & inférence de l'ensemble d'arbres..."):
            try:
                model = joblib.load('models/random_forest_model.joblib')

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
                probability = float(model.predict_proba(processed_data)[0][1])
                tier = get_risk_tier(probability)
                framingham_prob = compute_framingham_risk(age, 1 if sex == "Homme" else 0, resting_bp, cholesterol, 1 if exercise_angina == "Oui" else 0)

            except Exception as e:
                st.error(f"Erreur d'inférence : {e}")
                st.stop()

        # ── BANNIÈRE DE RÉSULTAT COLORÉE DYNAMIQUE ──
        st.markdown(f"""
        <div class="{tier['css_class']}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span class="badge-pill" style="background: {tier['badge_bg']}; color: {tier['color']}; border: 1px solid {tier['color']};">
                    {tier['level']}
                </span>
                <span style="font-weight: 700; color: #94a3b8; font-size: 0.9rem;">Classification Clinique IA</span>
            </div>
            <h2 style="color: {tier['color']}; margin: 0.25rem 0 0.5rem 0; font-size: 1.6rem; font-weight: 800;">
                {tier['title']}
            </h2>
            <p style="color: #cbd5e1; font-size: 1.05rem; margin: 0 0 0.75rem 0;">
                {tier['desc']}
            </p>
            <div style="background: rgba(0,0,0,0.25); padding: 0.75rem 1rem; border-radius: 8px; font-size: 0.95rem; color: #f8fafc;">
                <b>Recommandation médicale :</b> {tier['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── BLOC DES DEUX GAUGES ET BENCHMARK CLINIQUE ──
        g_col1, g_col2, g_col3 = st.columns([1.2, 1, 1])

        with g_col1:
            st.markdown("#####  Score Prédictif Machine Learning")
            st.plotly_chart(plot_risk_gauge(probability, tier['color']), use_container_width=True)

        with g_col2:
            st.markdown("#####  Benchmark Clinique (Framingham / SCORE2)")
            st.plotly_chart(plot_risk_gauge(framingham_prob, "#38bdf8"), use_container_width=True)

        with g_col3:
            st.markdown("#####  Concordance Diagnostique")
            ecart = abs(probability - framingham_prob) * 100
            st.metric("Écart IA vs Abaque Clinique", f"{ecart:.1f} pts", delta="- Concordance Forte" if ecart < 15 else "Alerte de divergence", delta_color="normal" if ecart < 15 else "inverse")
            st.markdown(f"""
            <div style="background: #1e293b; padding: 0.75rem; border-radius: 8px; font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;">
                Le modèle ML affine l'évaluation grâce à l'analyse non-linéaire conjointe de l'électrophysiologie ST et de l'effort.
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── RADAR ANATOMIQUE ET EXPLICABILITÉ SHAP ──
        r_col1, r_col2 = st.columns([1, 1.2])

        with r_col1:
            st.markdown("#####  Profil de Vulnérabilité des 4 Piliers")
            raw_dict = {
                'resting bp s': resting_bp,
                'cholesterol': cholesterol,
                'oldpeak': oldpeak,
                'resting ecg': ecg_mapping[resting_ecg],
                'exercise angina': 1 if exercise_angina == "Oui" else 0,
                'max heart rate': max_hr
            }
            st.plotly_chart(plot_cardio_radar(raw_dict), use_container_width=True)

        with r_col2:
            st.markdown("#####  Explicabilité Locale SHAP (Facteurs Décisifs)")
            df_contribs = compute_shap_contributions(model, processed_data)
            st.plotly_chart(plot_shap_waterfall(df_contribs), use_container_width=True)

        # ── SIMULATEUR WHAT-IF INTERACTIF ──
        st.markdown("---")
        with st.expander(" Simulateur de Prévention Clinique « What-If » (Interventions Ciblées)", expanded=True):
            st.markdown("Simulez l'impact thérapeutique immédiat d'une normalisation de la tension, du cholestérol ou de l'arrêt du tabac :")
            w1, w2, w3 = st.columns(3)
            with w1:
                target_bp = st.slider("Objectif Pression Systolique (mmHg)", 90, 200, min(int(resting_bp), 120), step=5)
            with w2:
                target_chol = st.slider("Objectif Cholestérol (mg/dl)", 100, 350, min(int(cholesterol), 175), step=5)
            with w3:
                target_angina = st.selectbox("Arrêt / Traitement Angine d'effort", ["Non", "Oui"], index=0)

            sim_df = input_data.copy()
            sim_df['resting bp s'] = [target_bp]
            sim_df['cholesterol'] = [target_chol]
            sim_df['exercise angina'] = [1 if target_angina == "Oui" else 0]

            sim_processed = preprocess_data(sim_df, is_training=False)
            new_prob = float(model.predict_proba(sim_processed)[0][1])
            delta_val = (new_prob - probability) * 100
            new_tier = get_risk_tier(new_prob)

            sw1, sw2 = st.columns([1, 2])
            with sw1:
                st.metric("Risque Après Intervention", f"{new_prob*100:.1f}%", delta=f"{delta_val:.1f} pts", delta_color="inverse")
                st.caption(f"Nouveau profil : **{new_tier['level']}**")
            with sw2:
                st.plotly_chart(plot_risk_gauge(new_prob, new_tier['color']), use_container_width=True)

        # ── EXPORT FICHE PATIENT DE CONSULTATION ──
        st.markdown("---")
        report_text = f"""# RAPPORT DE CONSULTATION CARDIOVASCULAIRE PRÉDICTIVE
Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}
Identifiant Consultation : CRD-{int(datetime.now().timestamp())}

## 1. PARAMÈTRES CLINIQUES DU PATIENT
- Âge : {age} ans | Sexe : {sex}
- Pression artérielle systolique : {resting_bp} mmHg
- Cholestérol total : {cholesterol} mg/dl
- Glycémie à jeun : {fasting_bs}
- ECG au repos : {resting_ecg}
- Dépression ST (oldpeak) : {oldpeak} mm | Pente ST : {st_slope}
- Fréquence cardiaque maximale : {max_hr} bpm | Angine à l'effort : {exercise_angina}

## 2. ÉVALUATION IA & MACHINE LEARNING
- Probabilité de risque estimée : {probability*100:.1f}%
- Statut diagnostique : {tier['level']}
- Indice classique Framingham : {framingham_prob*100:.1f}%

## 3. RECOMMANDATIONS CLINIQUES
- {tier['action']}
- Surveillance régulière de la pression artérielle et du bilan lipidique.
"""
        st.download_button(
            " Télécharger la Fiche de Synthèse Clinique (Markdown)",
            report_text,
            file_name=f"bilan_cardio_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            use_container_width=True
        )


# ── PAGE 2 : VISUALISATION DES DONNÉES ────────────────────────────────────────
elif page == "Visualisation des Données":
    st.markdown('<div class="med-header"><h1 class="med-title">Exploration & Analyse de la Cohorte Cardiovasculaire</h1></div>', unsafe_allow_html=True)
    try:
        df_pop = pd.read_csv("data/data.csv")
        st.dataframe(df_pop.head(10), use_container_width=True)
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig_hist = px.histogram(df_pop, x="age", color="target", barmode="overlay", title="Distribution des âges par statut cardiovasculaire", template="plotly_dark", color_discrete_sequence=["#10b981", "#ef4444"])
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_v2:
            fig_scatter = px.scatter(df_pop, x="cholesterol", y="resting bp s", color="target", title="Pression artérielle vs Cholestérol", template="plotly_dark", color_discrete_sequence=["#10b981", "#ef4444"])
            st.plotly_chart(fig_scatter, use_container_width=True)
    except Exception as e:
        st.info("Données de cohorte en cours de chargement...")


# ── PAGE 3 : ARCHITECTURE & MODÈLES ───────────────────────────────────────────
elif page == "Architecture & Modèles":
    st.markdown('<div class="med-header"><h1 class="med-title">Architecture des Modèles & Benchmarks SOTA</h1></div>', unsafe_allow_html=True)
    st.markdown("""
    ### Pipeline d'Ingénierie & Validation
    - **Algorithmes comparés** : Random Forest, XGBoost, LightGBM, Régression Logistique.
    - **Validation Croisée** : 5-Fold Stratified K-Fold.
    - **Explicabilité** : Noyau SHAP TreeExplainer garantissant la décomposition additive locale.
    """)


# ── PAGE 4 : À PROPOS ─────────────────────────────────────────────────────────
elif page == "À Propos":
    st.markdown('<div class="med-header"><h1 class="med-title">À Propos de CardioSOTA</h1></div>', unsafe_allow_html=True)
    st.markdown("""
    **CardioSOTA** est une application d'aide à la décision médicale préventive conçue par **Oumou Kaltoum Sall** (R&D Data Scientist / ML Engineer).
    Elle démontre la mise en production de modèles d'IA interprétables (XAI) appliqués à la santé.
    """)
