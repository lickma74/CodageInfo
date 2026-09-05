## Contexte du projet

Je dois développer un prototype d’algorithme de traitement numérique du signal dans le cadre d’un projet universitaire.

Le contexte est le suivant :

Un plongeur respire un mélange gazeux dans lequel l’azote est remplacé par de l’hélium. Cela permet d’éviter certains problèmes liés à la dissolution de l’azote dans le sang sous pression, mais cela rend la voix du plongeur difficilement compréhensible.

L’hélium augmente la vitesse de propagation des ondes sonores. Pour le signal vocal, cela se traduit par une **dilatation de l’enveloppe spectrale**. Le facteur de dilatation dépend de la profondeur et se situe typiquement entre **2 et 3**.

Cependant, la fréquence de vibration des cordes vocales du plongeur n’est pas significativement modifiée, puisqu’elle est contrôlée par le plongeur. Les fréquences fondamentales et donc la **position des harmoniques dans le spectre** restent donc essentiellement les mêmes.

Le but est donc de développer un algorithme capable de prendre une voix enregistrée sous hélium et de produire une version « restaurée », plus proche de la voix originale, en **compressant l’enveloppe spectrale tout en conservant la structure spectrale fine, notamment la position des harmoniques**.

---

## Objectif général

Je dois implémenter et comparer deux approches différentes :

### Approche 1 — LPC

Utiliser une modélisation de l’enveloppe spectrale basée sur la **prédiction linéaire (LPC)**.

L’analyse doit permettre d’obtenir notamment :

* les coefficients du filtre LPC ;
* le signal d’excitation ;
* les paramètres nécessaires à la synthèse du signal.

Ces paramètres doivent ensuite être modifiés afin de comprimer l’enveloppe spectrale d’un facteur compris entre environ **2 et 3**, sans déplacer les harmoniques.

Le signal restauré doit ensuite être reconstruit à partir du modèle LPC modifié.

### Approche 2 — FFT / DFT

Utiliser une approche basée sur une **décomposition fréquentielle**, notamment la DFT/FFT.

L’objectif est d’analyser le spectre du signal, de modifier son enveloppe spectrale afin de la comprimer d’un facteur de 2 à 3, tout en conservant la structure spectrale fine et donc les positions relatives des harmoniques, puis de reconstruire le signal temporel.

---

## Contraintes du projet

Le fichier d'entrée est :

* un fichier WAV ;
* mono ;
* fréquence d'échantillonnage : **44,1 kHz**.

Le fichier de sortie doit être un fichier WAV contenant le signal vocal restauré.

Il ne faut **pas** développer une application temps réel. Le traitement peut simplement fonctionner sous forme :

`input.wav → algorithme → output.wav`

La durée des trames doit être déterminée en fonction de la stationnarité de la parole, mais elle doit respecter :

* **maximum 50 ms par trame**.

On suppose que les paramètres du modèle restent constants pendant une trame, mais peuvent varier significativement d’une trame à l’autre.

Le traitement doit également respecter une contrainte sur l’amplitude :

* le signal de sortie doit avoir un ordre de grandeur d’amplitude comparable au signal d’entrée ;
* il ne doit pas y avoir de gain ou d’atténuation globale artificielle importante.

Je dois également justifier :

* la durée des trames choisie ;
* le type de fenêtre temporelle utilisé ;
* où la fenêtre doit être appliquée ;
* si un recouvrement est nécessaire ;
* le pourcentage de recouvrement choisi ;
* pourquoi ces choix sont appropriés au traitement de la parole.

---

# Ce que je veux que tu développes

Je veux que tu m’accompagnes dans l’implémentation complète du projet en **Python**.

Utilise autant que possible des bibliothèques standards du traitement du signal telles que :

* NumPy ;
* SciPy ;
* éventuellement Matplotlib pour visualiser les résultats ;
* éventuellement librosa si réellement pertinent.

Évite les abstractions inutiles et les fonctions « magiques » qui cachent complètement le traitement DSP. Je veux comprendre ce que fait réellement l'algorithme.

L'implémentation doit être suffisamment propre pour que je puisse la présenter et l'expliquer dans un contexte universitaire.

---

# IMPORTANT — Walkthrough pédagogique

Je ne veux PAS que tu me donnes immédiatement tout le code final.

Je veux que tu construises le projet **étape par étape**, avec un walkthrough au fur et à mesure de l'implémentation.

À chaque étape :

1. Explique d'abord **le problème que l'étape cherche à résoudre**.
2. Explique ensuite **le concept de traitement numérique du signal utilisé**.
3. Explique **pourquoi cette méthode est appropriée au problème**.
4. Explique les choix de paramètres et les alternatives possibles.
5. Donne ensuite le code correspondant à cette étape.
6. Explique le code bloc par bloc.
7. Indique ce que je devrais observer lorsque j'exécute le code.
8. Si possible, propose une visualisation permettant de vérifier que l'étape fonctionne.
9. Attends ensuite ma confirmation avant de passer à l'étape suivante.

Je veux donc apprendre le fonctionnement de l'algorithme pendant que nous le construisons, et pas simplement obtenir une solution finale.

---

# Ordre de développement souhaité

Je voudrais idéalement procéder dans cet ordre :

### Étape 1 — Analyse du signal d'entrée

Commencer par charger le WAV et vérifier :

* fréquence d'échantillonnage ;
* nombre de canaux ;
* durée ;
* amplitude ;
* éventuelle normalisation.

Afficher également quelques représentations de base :

* signal temporel ;
* spectrogramme ;
* spectre d’une portion du signal.

Le but est de comprendre concrètement le signal sur lequel nous travaillons.

### Étape 2 — Analyse de la stationnarité et choix des trames

Déterminer une durée de trame raisonnable pour de la parole.

Comparer éventuellement plusieurs valeurs comme 20, 30, 40 et 50 ms.

Expliquer le compromis entre :

* résolution temporelle ;
* résolution fréquentielle ;
* hypothèse de stationnarité ;
* coût computationnel.

Choisir ensuite une durée et un recouvrement justifiés.

### Étape 3 — Fenêtrage et segmentation

Implémenter le découpage en trames et le fenêtrage.

Expliquer notamment :

* pourquoi une fenêtre est nécessaire ;
* les effets de la troncature temporelle ;
* pourquoi choisir par exemple une fenêtre Hann/Hamming ou une autre fenêtre ;
* pourquoi utiliser ou non un recouvrement de 50 %.

### Étape 4 — Approche LPC

Implémenter progressivement :

1. calcul de l'autocorrélation ou méthode équivalente ;
2. estimation des coefficients LPC ;
3. représentation du filtre vocal ;
4. extraction/reconstruction de l'excitation ;
5. visualisation de l'enveloppe spectrale.

Je veux particulièrement comprendre le lien entre :

`signal vocal → LPC → enveloppe spectrale`

et pourquoi les coefficients LPC permettent de représenter l'enveloppe plutôt que les harmoniques individuelles.

### Étape 5 — Compression de l'enveloppe spectrale avec LPC

C'est une partie centrale du projet.

Déterminer précisément comment transformer le modèle LPC afin de :

* comprimer l'enveloppe spectrale d'un facteur donné ;
* conserver les fréquences des harmoniques ;
* éviter de modifier inutilement la fréquence fondamentale.

Explique mathématiquement et intuitivement ce qui est modifié dans le modèle.

Il faut également traiter le cas où le facteur de compression est variable, par exemple :

`compression_factor = 2.0`

puis

`compression_factor = 2.5`

et

`compression_factor = 3.0`.

### Étape 6 — Synthèse LPC

Reconstruire le signal à partir :

* de l'excitation ;
* du filtre LPC modifié.

Puis reconstruire le signal complet à partir des différentes trames.

Faire attention au problème de **overlap-add** et aux éventuelles discontinuités entre les trames.

### Étape 7 — Approche FFT

Implémenter une deuxième méthode indépendante basée sur la FFT.

Je veux comprendre comment séparer conceptuellement :

* l'enveloppe spectrale ;
* la structure fine du spectre ;
* les harmoniques.

Déterminer une méthode permettant de comprimer l'enveloppe spectrale sans simplement déplacer toutes les composantes fréquentielles.

### Étape 8 — Reconstruction FFT

Reconstruire le signal temporel à partir du spectre modifié en respectant notamment :

* magnitude ;
* phase ;
* symétrie du spectre pour un signal réel ;
* IFFT ;
* overlap-add.

Expliquer clairement le rôle de la phase et pourquoi elle ne doit pas être traitée de manière naïve.

### Étape 9 — Comparaison LPC vs FFT

Comparer les deux approches selon plusieurs critères :

* qualité perceptuelle ;
* intelligibilité ;
* fidélité de la voix ;
* conservation des harmoniques ;
* artefacts ;
* robustesse ;
* complexité computationnelle ;
* facilité d'implémentation ;
* comportement selon différents facteurs de compression.

Si possible, créer des visualisations permettant de comparer :

`signal original / signal hélium / signal restauré`

ainsi que leurs spectres ou spectrogrammes.

### Étape 10 — Normalisation et export

S'assurer que le signal de sortie :

* ne subit pas un gain global excessif ;
* reste dans une plage valide pour un fichier WAV ;
* conserve un niveau d'amplitude comparable à l'entrée.

Exporter finalement :

* `output_lpc.wav`
* `output_fft.wav`

---

# Point très important sur la modélisation

Ne fais pas simplement une opération naïve du type :

`f_new = f_old / 2`

sur toutes les fréquences du spectre.

Cela déplacerait également les harmoniques, ce qui est contraire à l'objectif du projet.

Le problème doit être traité comme une **compression de l'enveloppe spectrale autour d'une structure harmonique qui doit rester en place**.

Je veux que tu sois particulièrement attentif à cette distinction :

* **enveloppe spectrale** = forme globale du spectre ;
* **structure fine** = harmoniques et détails locaux.

L'objectif est de modifier la première sans détruire la seconde.

---

# Validation

À chaque étape importante, propose des tests permettant de vérifier que l'implémentation est correcte.

Par exemple :

* comparaison avant/après d'un spectre ;
* comparaison des positions des pics harmoniques ;
* comparaison de l'enveloppe spectrale ;
* comparaison temporelle ;
* écoute du fichier produit ;
* vérification de l'amplitude RMS/peak ;
* vérification de l'absence de clipping.

Si une approche est mathématiquement élégante mais produit des artefacts importants en pratique, explique pourquoi et propose une amélioration.

---

# Niveau d'explication

Je suis étudiant en génie informatique et j'ai des bases en traitement numérique du signal, mais je veux comprendre **le raisonnement derrière chaque décision**.

N'hésite donc pas à expliquer les mathématiques lorsqu'elles sont nécessaires, notamment :

* DFT/FFT ;
* fréquence fondamentale et harmoniques ;
* enveloppe spectrale ;
* LPC ;
* fonction de transfert du filtre ;
* excitation ;
* fenêtrage ;
* overlap-add ;
* résolution fréquentielle ;
* stationnarité.

Cependant, évite de partir dans des démonstrations mathématiques inutiles si elles n'aident pas directement à comprendre l'implémentation.

---

# Format de collaboration

Commence uniquement par me donner :

1. une vue d'ensemble de l'architecture du projet ;
2. les choix techniques que tu recommandes ;
3. les principaux défis que tu anticipes ;
4. l'Étape 1 avec son explication et son code.

**Ne donne pas tout le projet d'un seul coup.**

Après chaque étape, attends ma réponse avant de continuer afin que je puisse poser des questions et comprendre correctement le fonctionnement.

Si tu constates qu'une partie de mon énoncé est ambiguë ou qu'une décision technique nécessite une hypothèse, indique clairement cette hypothèse et explique ses conséquences plutôt que de la cacher dans le code.
