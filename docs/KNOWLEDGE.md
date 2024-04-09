# Connaissances utiles

## Appris et utile pour le projet

- détection du début des notes
  - filtre moyenneur sur l'énergie instantannée
  - filtre dérviateur (passe haut)
  - attaques d'énergies pour les notes
- convolution avec une suite h telle que $h_n = 0$ si $n < 0$ (filtre causal)
- création de filtres bidimensionnels à partri de filtre uni-dimensionnels grâce au produit tensoriel

## Appris et non utilisé

- TF sur Z: sert à étudier et concevoir des filtres
- TFCT: sert à étudier le signal avec une représentation temps/fréquence
- TF en 2D: sert à filtrer la TFCT

## Points du cours difficile

- pour certaines notes, la fondamentale n'est pas l'harmonique avec le plus d'énergie
- schéma fréquentiel des percussions
- l'influence des pôles et des zéros sur la réponse en fréquence
- réglage du régime transitoire du filtre
