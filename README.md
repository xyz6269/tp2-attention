# Rapport de TP : Image Captioning avec Attention (ResNet50 + LSTM)

**Sujet :** Génération automatique de légendes pour des images.
**Outils utilisés :** PyTorch, ResNet50 (Transfer Learning), Word2Vec, Mécanisme d'Attention.

---

## 1. Introduction et Objectifs
L'objectif de ce TP est de concevoir un système capable de générer une description textuelle pertinente à partir d'une image d'entrée. Ce projet combine la vision par ordinateur (pour comprendre l'image) et le traitement du langage naturel (pour générer la phrase).

**Points clés :**
* Utilisation du dataset **Flickr30k**.
* Extraction de caractéristiques visuelles via un **ResNet50** pré-entraîné.
* Utilisation d'**embeddings Word2Vec** pour la représentation des mots.
* Implémentation d'un **mécanisme d'attention spatial** pour focaliser le modèle sur des zones spécifiques de l'image lors de la génération de chaque mot.

---

## 2. Architecture du Modèle
Le modèle repose sur une architecture "Encoder-Decoder" améliorée par l'attention.

### A. L'Encodeur (Vision)
Nous utilisons un **ResNet50** dont les couches finales de classification ont été retirées. Le modèle est "gelé" (frozen) pour servir d'extracteur de caractéristiques fixes. La sortie utilisée est une carte de caractéristiques (feature map) provenant de la dernière couche convolutionnelle.

### B. Le Mécanisme d'Attention
Au lieu d'envoyer toute l'image d'un coup au décodeur, le module d'attention calcule des scores d'importance pour chaque région de l'image en fonction de l'état caché actuel du LSTM.



### C. Le Décodeur (Langage)
Il s'agit d'un **LSTM modifié**. À chaque étape temporelle $t$ :
1. Le mécanisme d'attention produit un "vecteur de contexte" à partir des caractéristiques de l'image.
2. Le LSTM reçoit en entrée la concaténation de l'embedding du mot précédent et de ce vecteur de contexte.
3. Il prédit le mot suivant dans la séquence.

---

## 3. Prétraitement des Données
* **Images :** Redimensionnement à 224x224 et normalisation selon les standards ImageNet.
* **Texte :** Tokenisation des légendes, ajout des balises `<start>` et `<end>`, et création d'un vocabulaire.
* **Embeddings :** Chargement de vecteurs Word2Vec pré-entraînés pour initialiser la couche d'embedding, permettant au modèle de bénéficier d'une compréhension sémantique préalable des mots.

---

## 4. Entraînement et Résultats
L'entraînement a été configuré avec un optimiseur **Adam** et un **StepLR** pour réduire le taux d'apprentissage progressivement.

**Performances observées :**
D'après les logs du notebook, le modèle atteint une **Perplexity de 40.12** à l'époque 7, avec une baisse constante de la perte de validation.

**Exemple de génération :**
* **Réel :** *The man with pierced ears is wearing glasses and an orange hat.*
* **Généré :** *a man with a beard and a gray shirt is holding a bottle of paper in a dimly lit*
* **Analyse :** Le modèle identifie correctement le sujet principal ("a man"), bien que certains détails spécifiques (couleur du chapeau) puissent être confondus ou simplifiés.

---

## 5. Conclusion
Ce TP démontre la puissance des mécanismes d'attention. Contrairement à un encodeur standard qui compresse l'image en un seul vecteur, l'attention permet au décodeur de "regarder" différentes parties de l'image dynamiquement. Cela améliore considérablement la précision des légendes pour des scènes complexes.

**Perspectives :**
* Dégeler les dernières couches du ResNet (Fine-tuning) pour adapter l'extraction de traits au dataset spécifique.
* Augmenter le nombre d'époques pour affiner la précision des détails (couleurs, objets secondaires).