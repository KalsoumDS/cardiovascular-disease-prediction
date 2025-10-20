# 🤝 Guide de Contribution

Merci de votre intérêt pour contribuer à ce projet de prédiction des maladies cardiovasculaires ! 

## 📋 Table des Matières

- [🚀 Démarrage Rapide](#-démarrage-rapide)
- [🔧 Configuration du Développement](#-configuration-du-développement)
- [📝 Types de Contributions](#-types-de-contributions)
- [🎯 Standards de Code](#-standards-de-code)
- [📋 Processus de Contribution](#-processus-de-contribution)
- [🐛 Signaler un Bug](#-signaler-un-bug)
- [💡 Proposer une Fonctionnalité](#-proposer-une-fonctionnalité)
- [📚 Documentation](#-documentation)
- [❓ Questions](#-questions)

## 🚀 Démarrage Rapide

1. **Fork** le repository
2. **Clone** votre fork localement
3. **Créez** une branche pour votre contribution
4. **Développez** votre fonctionnalité
5. **Testez** vos changements
6. **Soumettez** une Pull Request

## 🔧 Configuration du Développement

### Prérequis
- Python 3.8+
- Git
- Un éditeur de code (VS Code, PyCharm, etc.)

### Installation
```bash
# 1. Fork et clone le repository
git clone https://github.com/votre-username/cardiovascular-disease-prediction.git
cd cardiovascular-disease-prediction

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Tester l'installation
streamlit run app.py
```

## 📝 Types de Contributions

### 🐛 Corrections de Bugs
- Corriger des erreurs dans le code
- Améliorer la gestion d'erreurs
- Optimiser les performances

### ✨ Nouvelles Fonctionnalités
- Ajouter de nouveaux modèles ML
- Améliorer l'interface utilisateur
- Ajouter de nouvelles visualisations

### 📚 Documentation
- Améliorer le README
- Ajouter des commentaires dans le code
- Créer des tutoriels

### 🧪 Tests
- Ajouter des tests unitaires
- Améliorer la couverture de tests
- Tests d'intégration

## 🎯 Standards de Code

### Style de Code
- Suivre PEP 8 pour Python
- Utiliser des noms de variables descriptifs
- Commenter le code complexe

### Structure des Commits
```
type(scope): description courte

Description plus détaillée si nécessaire

Fixes #issue_number
```

Types de commits :
- `feat`: nouvelle fonctionnalité
- `fix`: correction de bug
- `docs`: documentation
- `style`: formatage
- `refactor`: refactoring
- `test`: tests
- `chore`: tâches de maintenance

### Exemple
```
feat(prediction): add new neural network model

- Implemented a simple neural network using TensorFlow
- Added model evaluation metrics
- Updated documentation

Closes #15
```

## 📋 Processus de Contribution

### 1. Créer une Branche
```bash
git checkout -b feature/nom-de-votre-fonctionnalite
# ou
git checkout -b fix/description-du-bug
```

### 2. Développer
- Écrivez du code propre et bien commenté
- Testez vos changements
- Mettez à jour la documentation si nécessaire

### 3. Tester
```bash
# Lancer l'application
streamlit run app.py

# Vérifier que tout fonctionne
python -m pytest tests/  # Si des tests existent
```

### 4. Commiter
```bash
git add .
git commit -m "feat: add new feature description"
```

### 5. Pousser
```bash
git push origin feature/nom-de-votre-fonctionnalite
```

### 6. Pull Request
- Créez une Pull Request sur GitHub
- Décrivez clairement vos changements
- Référencez les issues concernées

## 🐛 Signaler un Bug

### Avant de Signaler
1. Vérifiez que le bug n'a pas déjà été signalé
2. Testez avec la dernière version
3. Vérifiez la documentation

### Template de Bug Report
```markdown
**Description du Bug**
Une description claire du problème.

**Étapes pour Reproduire**
1. Aller à '...'
2. Cliquer sur '...'
3. Voir l'erreur

**Comportement Attendu**
Ce qui devrait se passer.

**Captures d'Écran**
Si applicable, ajoutez des captures d'écran.

**Environnement**
- OS: [ex: macOS, Windows, Linux]
- Python: [ex: 3.8.5]
- Streamlit: [ex: 1.27.0]

**Informations Supplémentaires**
Toute autre information pertinente.
```

## 💡 Proposer une Fonctionnalité

### Template de Feature Request
```markdown
**Fonctionnalité Demandée**
Description claire de la fonctionnalité.

**Problème à Résoudre**
Quel problème cette fonctionnalité résoudrait-elle ?

**Solution Proposée**
Description de la solution que vous aimeriez voir.

**Alternatives Considérées**
Autres solutions que vous avez considérées.

**Contexte Supplémentaire**
Toute autre information pertinente.
```

## 📚 Documentation

### Améliorer la Documentation
- Corriger les erreurs de frappe
- Ajouter des exemples
- Clarifier les instructions
- Traduire en d'autres langues

### Structure de la Documentation
```
docs/
├── installation.md
├── usage.md
├── api.md
├── contributing.md
└── examples/
    ├── basic_usage.py
    └── advanced_features.py
```

## ❓ Questions

### Où Poser des Questions
- **Issues GitHub** : Pour les bugs et fonctionnalités
- **Discussions GitHub** : Pour les questions générales
- **Email** : Pour les questions privées

### Types de Questions
- Questions sur l'installation
- Questions sur l'utilisation
- Questions sur le développement
- Questions sur la contribution

## 🏆 Reconnaissance

Tous les contributeurs seront mentionnés dans :
- Le fichier CONTRIBUTORS.md
- Les notes de version
- La documentation

## 📞 Contact

- **Maintainer** : [Votre Nom]
- **Email** : [votre.email@example.com]
- **GitHub** : [@votre-username]

---

Merci de contribuer à ce projet ! Votre aide est précieuse pour améliorer la prédiction des maladies cardiovasculaires. 🫀❤️
