# Briefings communs en classe

## Difficultés

- Comment découper temporellement le signal audio pour séparer les notes
- Quelle est la durée de la note (si la note ne se termine pas de facon abrupte, comment sait-on lorsqu'elle est finie)
- Comment gérer les accords
- Comment différencier une jouée des harmoniques des notes plus graves
- Principe d'incertitude de Heisenberg (plus on est précis temporellement, moins on est précis fréquentiellement et vice versa)
- Plusieurs instruments peuvent jouer la même note
- Plusieurs notes peuvent être jouées en même temps

## Connaissances nécessaires

- Lien entre harmonique et le type d'instrument
- TF pour analyser une note connue
- TF à court terme !
- Filtrage !

## Livrable

- **Entrée :** signal audio
- **Sortie :** tableau de booléens avec les notes et instruments

| Clef       | t0   | t1    | ... | tn    |
| ---------- | ---- | ----- | --- | ----- |
| do         | true | false | ... | false |
| re         | true | true  | ... | false |
| ...        | ...  | ...   | ... | ...   |
| si         | true | true  | ... | false |
| guitare    | true | true  | ... | false |
| percussion | true | true  | ... | false |

> **Notes :**
>
> Les notes sont uniquements pour la guitare, pas pour les percussions.
>
> Il n'y a pas d'autres instruments.
