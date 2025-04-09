# Face_Id_Streamlit
# https://github.com/youssou91/Face_Id_Streamlit

Cas d'utilisation(Issus des exigences fonctionnelles)
d) Cas d'utilisation de l'Éditeur
• Éditer une description d’ouvrage : Rédaction de la description et affectation d’un
niveau d’expertise (débutant, amateur, chef).
• Gérer les catégories d’ouvrages : Ajout, modification, ou suppression des
catégories.
• Valider les commentaires : Consultation des commentaires en attente et validation
ou rejet via une simple action.
e) Cas d'utilisation du Gestionnaire
(Détails à développer ultérieurement)
• Gérer le catalogue : Ajout, modification, suppression d’ouvrages, et consultation du
catalogue.
• Gérer le stock : Ajout et retrait d’articles du stock, recherche par niveau de stock.
• Suivre les ventes : Analyse des ventes selon divers critères (période, niveau de
détail).
f) Cas d'utilisation de l’Administrateur
(Détails à développer ultérieurement)
• Maintenir le site : Gestion des utilisateurs du back office (éditeurs, gestionnaires,
administrateurs) et des comptes clients.
Travail demandé
1. Modélisation des Cas d’Utilisation et du Diagramme de Classes
À partir de l’analyse des besoins fournie, vous devez :
1. Identifier et modéliser un diagramme de cas d’utilisation du back office en tenant
compte des rôles suivants :
➢ Éditeur (gestion des descriptions et des catégories d’ouvrages, validation des
commentaires).
➢ Gestionnaire (gestion du catalogue, gestion du stock, suivi des ventes).
➢ Administrateur (gestion des utilisateurs et maintenance du site).
2. Déduire un diagramme de classes détaillé en prenant en compte les entités du back
office et leurs relations.
Première ébauche du diagramme de cas d’utilisation
2. Personnalisation d’un Template de Dashboard
Vous devrez personnaliser un template de dashboard existant basé sur un framework
comme Bootstrap (https://themewagon.com/) et l’adapter aux besoins du projet.
L’interface doit inclure :
➢ Un tableau de bord principal avec des indicateurs clés (ventes, stock, nouveaux
commentaires, etc.).
➢ Des interfaces spécifiques permettant d’exécuter les actions définies dans les cas
d’utilisation.
➢ Un système de gestion des rôles pour limiter l’accès aux différentes sections en
fonction du type d’utilisateur.
3. Développement d’une API RESTful
Afin de permettre la communication entre le dashboard et la base de données du site ecommerce, vous devez :
• Concevoir une API RESTful permettant la gestion du catalogue, des stocks, des
ventes et des commentaires.
• Définir les endpoints nécessaires en fonction des besoins des utilisateurs du back
office.
• Tester l’API avec un outil tel que Postman et documenter son utilisation.