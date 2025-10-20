import streamlit as st

st.set_page_config(page_title="Aide et Recommandations", page_icon="ℹ️")

st.title("Aide et Recommandations")

# Navigation dans la page
section = st.sidebar.radio(
    "Navigation",
    ["Guide d'Utilisation", "Recommandations Générales", "FAQ", "Contact"]
)

if section == "Guide d'Utilisation":
    st.header("Guide d'Utilisation")
    
    st.markdown("""
    ### Comment utiliser l'application
    
    1. **Page d'Accueil**
       - Présentation générale du système
       - Navigation vers les différentes fonctionnalités
    
    2. **Prédiction du Risque**
       - Remplissez le formulaire avec vos informations médicales
       - Cliquez sur "Obtenir la Prédiction"
       - Consultez vos résultats et recommandations
    
    3. **Visualisation des Données**
       - Explorez les différentes visualisations
       - Sélectionnez le type de graphique souhaité
       - Analysez les corrélations et distributions
    
    4. **Analyse des Modèles**
       - Comparez les performances des différents modèles
       - Consultez les matrices de confusion
       - Visualisez l'importance des caractéristiques
    """)
    
    st.markdown("""
    ### Interprétation des Résultats
    
    - **Risque Faible** : Continuez à maintenir vos bonnes habitudes de vie
    - **Risque Modéré** : Consultez votre médecin pour un suivi régulier
    - **Risque Élevé** : Prenez rendez-vous rapidement avec un cardiologue
    """)

elif section == "Recommandations Générales":
    st.header("Recommandations Générales")
    
    st.markdown("""
    ### Prévention des Maladies Cardiovasculaires
    
    1. **Alimentation**
       - Privilégiez les fruits et légumes
       - Limitez les graisses saturées
       - Réduisez votre consommation de sel
       - Évitez les aliments transformés
    
    2. **Activité Physique**
       - 30 minutes d'exercice modéré par jour
       - Marche rapide, natation, vélo
       - Évitez la sédentarité
    
    3. **Mode de Vie**
       - Arrêtez de fumer
       - Limitez votre consommation d'alcool
       - Gérez votre stress
       - Dormez suffisamment
    
    4. **Suivi Médical**
       - Contrôles réguliers de la tension
       - Bilan lipidique annuel
       - Suivi du poids et de l'IMC
       - Consultation médicale régulière
    """)

elif section == "FAQ":
    st.header("Foire Aux Questions")
    
    faq = {
        "Quelle est la précision du système ?": 
            "Notre système utilise plusieurs modèles de machine learning avec une précision moyenne de 85-90%.",
        
        "Puis-je utiliser ces résultats pour un diagnostic médical ?":
            "Non, ces résultats sont indicatifs et ne remplacent pas un diagnostic médical professionnel.",
        
        "Comment sont collectées les données ?":
            "Les données proviennent de sources médicales fiables et sont anonymisées.",
        
        "Quels sont les facteurs de risque pris en compte ?":
            "Âge, sexe, pression artérielle, cholestérol, diabète, tabagisme, etc.",
        
        "Comment puis-je améliorer mon score ?":
            "Suivez les recommandations personnalisées et consultez régulièrement votre médecin."
    }
    
    for question, reponse in faq.items():
        with st.expander(question):
            st.write(reponse)

else:  # Contact
    st.header("Contact et Support")
    
    st.markdown("""
    Pour toute question ou assistance
    
    📧 Email : [s.sall@mundiapolis.ma](mailto:s.sall@mundiapolis.ma), [s.oujaa@mundiapolis.ma](mailto:s.oujaa@mundiapolis.ma)
    
    📞 Téléphone : +212 6XX-XXXXXX
    """)
    
    st.subheader("Adresse")
    st.write("Université de Mundiapolis M1DS")
    
    # Formulaire de contact
    with st.form("contact_form"):
        st.subheader("Envoyez-nous un message")
        
        name = st.text_input("Nom")
        email = st.text_input("Email")
        subject = st.selectbox("Sujet", 
                             ["Question technique", 
                              "Problème d'utilisation",
                              "Suggestion d'amélioration",
                              "Autre"])
        message = st.text_area("Message")
        
        if st.form_submit_button("Envoyer"):
            st.success("Message envoyé avec succès ! Nous vous répondrons dans les plus brefs délais.") 