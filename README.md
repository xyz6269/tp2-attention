# Rapport Technique Approfondi : Captioning d'Images avec Attention

**Étudiants :** Boulaamail Mohamed Ali
**Sujet :** Génération de légendes via ResNet50 et LSTM à Attention  
**Dataset :** Flickr30k  

---

## 1. Analyse de l'Architecture Multi-Modale

Le projet repose sur la fusion de deux domaines du Deep Learning : la **Vision par Ordinateur (CV)** et le **Traitement du Langage Naturel (NLP)**.

### 1.1. L'Encodeur : ResNet50 et Feature Mapping
Au lieu d'utiliser un vecteur global (Global Average Pooling), nous avons extrait les cartes de caractéristiques de la dernière couche convolutionnelle de **ResNet50**. 
- **Format de sortie :** $7 \times 7$ avec 2048 canaux.
- **Pourquoi ?** Cela permet de conserver la structure spatiale de l'image. Le modèle ne voit pas seulement "un chien", il voit "un chien en haut à gauche".



### 1.2. Le Mécanisme d'Attention "Soft Attention"
C'est le cœur du TP. Pour chaque mot généré, le décodeur calcule un **score d'alignement** entre l'état caché du LSTM ($h_{t-1}$) et chaque zone du $7 \times 7$ de l'image.
- Ces scores sont passés dans une fonction **Softmax** pour obtenir des poids d'attention (somme = 1).
- Un **vecteur de contexte** est calculé par la somme pondérée des zones de l'image.



---

## 2. Implémentation du Décodeur LSTM

Le décodeur est un **LSTM modifié**. Contrairement à un LSTM classique qui ne prend que le mot précédent en entrée, celui-ci reçoit un "contexte visuel" dynamique à chaque étape $t$ :

$$x^{input}_t = [Embed(word_{t-1}) \ ; \ Context_{vison}]$$

L'utilisation d'**embeddings Word2Vec** pré-entraînés est cruciale ici : elle permet au modèle de comprendre que "man" et "boy" sont sémantiquement proches avant même d'avoir vu la première image du dataset.

---

## 3. Analyse des Performances et "Bad Results"

Dans le notebook, les résultats montrent une **Perplexity de 40.12** et des légendes qui, bien que syntaxiquement correctes, manquent de précision factuelle (ex: confusion sur les couleurs ou les objets secondaires).

### Pourquoi la performance est-elle limitée ?

1. **Le "Bottleneck" du ResNet Gelé :** En gelant (freezing) l'entièreté du ResNet, on force le décodeur à travailler avec des caractéristiques optimisées pour la classification ImageNet (1000 classes), et non pour la description détaillée. Un *Fine-tuning* des dernières couches améliorerait la détection des petits objets.

2. **La Taille du Vocabulaire vs Dataset :** Flickr30k est vaste, mais le modèle doit apprendre à la fois la grammaire anglaise et la reconnaissance visuelle. Avec seulement 7 à 10 époques (comme vu dans le notebook), le modèle est encore en phase de "sous-apprentissage" (*underfitting*).

3. **Le Phénomène de "Hallucination" :** Le modèle génère souvent des mots "sûrs" comme *a man in a shirt* car ils apparaissent statistiquement très souvent dans le dataset. C'est ce qu'on appelle le biais de fréquence : le modèle privilégie la probabilité linguistique sur l'observation visuelle réelle.

4. **L'Attention Non-Optimisée :** Si les poids d'attention ne sont pas bien entraînés, le modèle regarde "partout et nulle part" à la fois, produisant des descriptions génériques.

---

## 4. Évaluation Qualitative

**Exemple tiré du notebook :**
- **Réel :** *The man with pierced ears is wearing glasses and an orange hat.*
- **Généré :** *a man with a beard and a gray shirt is holding a bottle of paper...*

**Analyse de l'erreur :**
Le modèle a correctement identifié le sujet principal (**man**). Cependant, il a "halluciné" une barbe (**beard**) et une bouteille (**bottle**). Cela suggère que les poids d'attention se sont fixés sur des zones de pixels floues ou que le LSTM a sur-appris des séquences de mots communes au détriment de l'image.

---

## 5. Pistes d'Amélioration

Pour dépasser ces limitations, plusieurs stratégies sont envisageables :
* **Teacher Forcing Decay :** Réduire progressivement l'aide apportée au modèle pendant l'entraînement.
* **Beam Search :** Au lieu de prédire le mot le plus probable (Greedy), explorer plusieurs chemins de phrases possibles pour trouver la plus cohérente.
* **Data Augmentation :** Augmenter le nombre d'images par des rotations/crops pour forcer l'attention à être plus robuste.
* **Fine-tuning :** Libérer les poids du ResNet après quelques époques.