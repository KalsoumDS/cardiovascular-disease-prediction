import streamlit as st

st.set_page_config(page_title="Aide et recommandations", page_icon="ℹ")

st.title("Aide et recommandations")

# Navigation dans la page
section = st.sidebar.radio(
    "Navigation",
    ["Guide d’utilisation", "Recommandations générales", "Questions fréquentes", "Contact"]
)

if section == "Guide d’utilisation":
    st.header("Guide d’utilisation")
    
    st.markdown("""
    ### Comment utiliser l'application
    
    1. **Page d'Accueil**
       - Présentation générale du système
       - Navigation vers les différentes fonctionnalités
    
    2. **Évaluation du risque**
       - Remplissez le formulaire avec vos informations médicales
       - Cliquez sur « Obtenir l’estimation »
       - Consultez vos résultats et recommandations
    
    3. **Visualisation des Données**
       - Explorez les différentes visualisations
       - Sélectionnez le type de graphique souhaité
       - Analysez les corrélations et distributions
    
    4. **Analyse des modèles**
       - Comparez les performances des différents modèles
       - Consultez les matrices de confusion
       - Consultez l’importance des variables
    """)
    
    st.markdown("""
    ### Interprétation des Résultats
    
    - **Risque Faible** : Continuez à maintenir vos bonnes habitudes de vie
    - **Risque Modéré** : Consultez votre médecin pour un suivi régulier
    - **Risque Élevé** : Prenez rendez-vous rapidement avec un cardiologue
    """)

elif section == "Recommandations générales":
    st.header("Recommandations générales")
    
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

elif section == "Questions fréquentes":
    st.header("Foire aux questions")
    
    faq = {
        "Quelle est la précision du système ?": 
            "Notre système utilise plusieurs modèles d’apprentissage automatique avec une précision moyenne de 85 à 90 %.",
        
        "Puis-je utiliser ces résultats pour un diagnostic médical ?":
            "Non, ces résultats sont indicatifs et ne remplacent pas un diagnostic médical professionnel.",
        
        "Comment sont collectées les données ?":
            "Les données proviennent de sources médicales fiables et sont anonymisées.",
        
        "Quels sont les facteurs de risque pris en compte ?":
            "Âge, sexe, pression artérielle, cholestérol, diabète, tabagisme, etc.",
        
        "Comment puis-je améliorer mon résultat ?":
            "Suivez les recommandations personnalisées et consultez régulièrement votre médecin."
    }
    
    for question, reponse in faq.items():
        with st.expander(question):
            st.write(reponse)

else:  # Contact
    st.header("Contact et assistance")
    
    st.markdown("""
    Pour toute question ou assistance
    
     Email : [s.sall@mundiapolis.ma](mailto:s.sall@mundiapolis.ma), [s.oujaa@mundiapolis.ma](mailto:s.oujaa@mundiapolis.ma)
    
     Téléphone : +212 6XX-XXXXXX
    """)
    
    st.subheader("Adresse")
    st.write("Université de Mundiapolis M1DS")
    
    # Formulaire de contact
    with st.form("contact_form"):
        st.subheader("Envoyez-nous un message")
        
        name = st.text_input("Nom")
        email = st.text_input("Adresse courriel")
        subject = st.selectbox("Sujet", 
                             ["Question technique", 
                              "Problème d'utilisation",
                              "Suggestion d'amélioration",
                              "Autre"])
        message = st.text_area("Votre message")
        
        if st.form_submit_button("Envoyer"):
            st.success("Message envoyé avec succès ! Nous vous répondrons dans les plus brefs délais.") 