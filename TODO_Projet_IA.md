# 🐱🐶 Projet IA / Java / THS — Détection d'images (chat vs pas-chat)

> Checklist de suivi du projet. **Cochez la case** quand une tâche est terminée, puis *commit + pull*.
> Tout est à rendre le **05/06 avant 12h00** (devoir écrit 13h30, soutenance 14h45).

---

## 📖 Légende

| Symbole | Signification |
|--------|---------------|
| `[ ]` / `[x]` | À faire / Fait |
| 👥 **N pers** | Nombre de personnes qui peuvent travailler **en même temps** sur cette tâche |
| 🔗 **dép: …** | Tâche(s) à finir **avant** de pouvoir commencer celle-ci |
| ⚡ | **Parallélisable** : peut être faite en même temps que d'autres tâches du même bloc |
| 🚧 | **Chemin critique** : bloque la suite, à prioriser |
| 🧪 | Tâche d'expérimentation / mesure (alimente le rapport) |

**Codes des tâches** : `N#` = Nécessaire (obligatoire), `E#` = Extension (bonus).
Référez-vous à ces codes pour les dépendances et dans vos commits (ex : `git commit -m "N7 done"`).

> ⚠️ **Règle non négociable** : le code fourni (`iNeurone`, `Neurone`, `NeuroneHeaviside`, `testNeurone`, `Image`) **doit être utilisé tel quel**. On peut l'**étendre** (héritage), mais **pas le remplacer**. Sinon → 100 % de pénalité.

---

# ✅ PARTIE 1 — TÂCHES NÉCESSAIRES (obligatoires)

## Phase 0 — Mise en place 🚧
*À faire en tout premier. Bloque l'organisation de tout le monde.*

- [x] **N0.1** — Créer le dépôt GitHub + structure de dossiers + `.gitignore` (exclure `dataset_animaux/`) — 👥 **1 pers** · 🚧
- [x] **N0.2** — Récupérer le jeu de données du groupe (train + test) et le placer en local (PAS sur Git) — 👥 **1 pers** · ⚡
- [x] **N0.3** — Vérifier que tout le monde compile et exécute `testNeurone` sans erreur — 👥 **tous** · 🔗 dép: N0.1
- [x] **N0.4** — Nommer le chef de projet + se répartir les tâches ci-dessous — 👥 **tous**

---

## Phase 1 — Niveau 1 : maîtrise de l'objet neurone
*Le sujet l'exige explicitement. Indépendant du reste → peut tourner en parallèle de la Phase 3.*

- [x] **N1.1** — 🚧 Comprendre et **commenter ligne par ligne** l'algorithme d'apprentissage (`apprentissage()` dans `Neurone.java`, non commenté) : règle de mise à jour des poids, rôle de `eta`, du `delta`, du biais, condition d'arrêt — 👥 **1 pers** · ⚡
- [x] **N1.2** — 🧪 Lancer `testNeurone` sur la fonction **ET** (déjà codé) et **OU** : vérifier que ça apprend correctement — 👥 **1 pers** · 🔗 dép: N0.3 · ⚡
- [x] **N1.3** — 🧪 Lancer l'apprentissage **plusieurs fois** (ex : 20–50 runs) et collecter les valeurs finales des poids + biais — 👥 **1 pers** · 🔗 dép: N1.2
- [ ] **N1.4** — 🧪 Analyser : les poids trouvés sont-ils similaires d'un run à l'autre ? **Justifier** (init aléatoire, solutions multiples, séparabilité linéaire) — 👥 **1 pers** · 🔗 dép: N1.3

---

## Phase 2 — Niveau 1 : autres fonctions d'activation
*Les deux classes sont indépendantes → 2 personnes peuvent coder en parallèle. S'appuient sur la compréhension de `Neurone` (N1.1).*

- [x] **N2.1** — Créer la classe `NeuroneSigmoide extends Neurone` (activation = sigmoïde `1/(1+e^-x)`) — 👥 **1 pers** · 🔗 dép: N1.1 · ⚡
- [ ] **N2.2** — Créer la classe `NeuroneReLU extends Neurone` (activation = `max(0, x)`) — 👥 **1 pers** · 🔗 dép: N1.1 · ⚡
- [ ] **N2.3** — 🧪 Tester ET/OU avec Sigmoïde puis ReLU (décommenter les lignes prévues dans `testNeurone`) et comparer le comportement à Heaviside — 👥 **1 pers** · 🔗 dép: N2.1, N2.2
- [ ] **N2.4** — ⚠️ **Gérer le risque de boucle infinie** : la condition `while (mse > MSElimite)` peut ne jamais finir (problème non linéairement séparable, ReLU/sigmoïde qui ne converge pas). Ajouter un **nombre max d'itérations** via une sous-classe ou une surcharge (sans modifier `Neurone.java` directement) — 👥 **1 pers** · 🔗 dép: N1.1 · 🚧

> 💡 **N2.4 est important** : sans garde-fou, l'apprentissage sur les vraies images (Phase 4) risque de tourner à l'infini. À traiter avant la Phase 4.

---

## Phase 3 — Niveau 2 : brique données / images
*Cœur technique. Indépendant des Phases 1 et 2 → tourne en parallèle. C'est le plus gros morceau de code → 1 à 2 pers.*

- [ ] **N3.1** — 🚧 Étudier `Image.java` : comprendre que `donnees()` renvoie déjà l'image **aplatie en 1D** (= l'entrée directe d'un neurone) et que la conversion niveaux de gris est faite — 👥 **1 pers** · ⚡
- [ ] **N3.2** — Écrire la fonction qui **parcourt les dossiers** `train/` et crée la liste des `Image` (réutiliser `Image.listeFichiers`) — 👥 **1-2 pers** · 🔗 dép: N3.1 · 🚧
- [ ] **N3.3** — **Labelliser** chaque image selon le nom du dossier (`cat` → actif=1, sinon → 0) — 👥 **1 pers** · 🔗 dép: N3.2
- [ ] **N3.4** — **Normaliser** les amplitudes des pixels (ramener `[0..255]` → `[0..1]`, ou centrer-réduire) — 👥 **1 pers** · 🔗 dép: N3.3 · 🚧
- [ ] **N3.5** — **Mélanger** (shuffle) les données d'entraînement (attention à mélanger entrées **et** labels ensemble) — 👥 **1 pers** · 🔗 dép: N3.4
- [ ] **N3.6** — Construire les structures finales `float[][] entrees` + `float[] resultats` attendues par `apprentissage()` — 👥 **1 pers** · 🔗 dép: N3.5 · 🚧
- [ ] **N3.7** — Vérifier que **toutes les images ont la même dimension** (le neurone a un nombre de synapses fixe). Si non → prévoir un redimensionnement / filtrage — 👥 **1 pers** · 🔗 dép: N3.2

> ⚠️ **N3.7 est un piège classique** : si les images n'ont pas toutes la même taille, le `float[]` d'entrée change de longueur et le neurone plante. À vérifier tôt.

---

## Phase 4 — Niveau 2 : assemblage de la chaîne complète 🚧
*Le `main` qui relie tout. Dépend des Phases 2 et 3 → c'est le point de convergence du groupe.*

- [ ] **N4.1** — Écrire le `main` orchestrateur : train → label → normalise → mélange → entraîne — 👥 **1 pers** · 🔗 dép: N2.1 (ou Heaviside), N2.4, N3.6 · 🚧
- [ ] **N4.2** — **Entraîner** le neurone sur les données `train` (actif = chat) — 👥 **1 pers** · 🔗 dép: N4.1 · 🚧
- [ ] **N4.3** — Charger les données de **test**, les labelliser et normaliser de la **même façon** que le train — 👥 **1 pers** · 🔗 dép: N3.4
- [ ] **N4.4** — Appliquer le neurone entraîné sur le test (`metAJour` + lecture `sortie()` avec seuil) — 👥 **1 pers** · 🔗 dép: N4.2, N4.3 · 🚧
- [ ] **N4.5** — 🧪 Calculer le **taux de bonne classification** (et idéalement : vrais/faux positifs/négatifs) — 👥 **1 pers** · 🔗 dép: N4.4
- [ ] **N4.6** — (optionnel mais conseillé) Utiliser `sauvegarde()` / `chargement()` pour ne pas réentraîner à chaque fois — 👥 **1 pers** · 🔗 dép: N4.2 · ⚡

---

## Phase 5 — Rapport, soutenance & livrable (NOTÉ — obligatoire) 🚧
*40 % rapport+présentation, 40 % soutenance. À alimenter en continu, pas à la fin. 1 pers dédiée + apport de tous.*

- [ ] **N5.1** — Mettre en place la trame du **rapport technique PDF** (problématique → théorie → expérimentations → résultats → conclusion) — 👥 **1 pers** · ⚡
- [ ] **N5.2** — 🧪 Rédiger la section **maîtrise du neurone** (explication de l'algo N1.1 + stats sur les poids N1.4) — 👥 **1 pers** · 🔗 dép: N1.4
- [ ] **N5.3** — 🧪 Rédiger la section **résultats de classification** (chiffres de N4.5, analyse critique) — 👥 **1 pers** · 🔗 dép: N4.5
- [ ] **N5.4** — Rédiger le **manuel d'utilisation** (comment recompiler et lancer) — 👥 **1 pers** · 🔗 dép: N4.1
- [ ] **N5.5** — Préparer la **présentation PDF de soutenance** : mettre en avant **≥ 2 difficultés rencontrées** + comment résolues — 👥 **2 pers** · 🔗 dép: N5.2, N5.3
- [ ] **N5.6** — Page de garde + relecture + structure propre des deux PDF — 👥 **1 pers** · 🔗 dép: N5.5
- [ ] **N5.7** — 🚧 **Préparer l'envoi** : code source SANS dataset, SANS biblio ajoutée, **noms du groupe + numéro de groupe** dans le mail, numéro de groupe dans le nom du PDF de présentation — 👥 **1 pers** · 🔗 dép: tout
- [ ] **N5.8** — 🚧 **ENVOYER** aux 3 moniteurs avant l'heure limite (code avant 12h, présentation avant 11h59) — 👥 **chef de projet**

> ❗ **100 % de pénalité si le livrable est non-conforme.** Vérifier deux fois N5.7 avant N5.8.

---

# ⭐ PARTIE 2 — EXTENSIONS (par ordre de priorité)

> À faire **uniquement** une fois toute la Partie 1 terminée. Classées de la plus rentable/simple à la plus ambitieuse. Le Niveau 3 (recul critique) se gagne surtout ici.

## Priorité haute — gros impact, peu d'effort
- [ ] **E1** — 🧪 Comparer les **3 fonctions d'activation** (Heaviside / Sigmoïde / ReLU) sur le vrai problème chat/pas-chat → tableau de taux de réussite — 👥 **1 pers** · 🔗 dép: N4.5, N2.3 · ⚡
- [ ] **E2** — 🧪 Tester **sans normalisation** et comparer au résultat normalisé — 👥 **1 pers** · 🔗 dép: N4.5 · ⚡
- [ ] **E3** — 🧪 Tester **sans le mélange** des données et comparer — 👥 **1 pers** · 🔗 dép: N4.5 · ⚡
- [ ] **E4** — 🧪 Faire varier les **paramètres du neurone** (`eta`, `MSElimite`, nb d'itérations) et tracer l'effet sur la convergence/précision — 👥 **1 pers** · 🔗 dép: N4.5 · ⚡

## Priorité moyenne — démarche scientifique « signal »
- [ ] **E5** — 🧪 Robustesse au **bruit** sur ET/OU : ajouter un bruit d'amplitude contrôlée aux entrées, mesurer la dégradation, introduire la notion de **rapport signal/bruit** — 👥 **1 pers** · 🔗 dép: N1.2 · ⚡
- [ ] **E6** — Traiter les images en **couleur RGB** au lieu des niveaux de gris (`Image` le permet déjà) et comparer — 👥 **1 pers** · 🔗 dép: N4.5
- [ ] **E7** — Traiter en **couleur TSL/HSL** au lieu de RGB et comparer — 👥 **1 pers** · 🔗 dép: E6

## Priorité basse — ambitieux
- [ ] **E8** — Classifier en plus **chiens** et **animaux sauvages** (multi-classes : plusieurs neurones, un par catégorie) — 👥 **1-2 pers** · 🔗 dép: N4.5
- [ ] **E9** — **Augmentation de données** : miroir, égalisation d'histogramme, etc. — 👥 **1 pers** · 🔗 dép: N3.6
- [ ] **E10** — Appliquer une **FFT 2D** sur les images avant le neurone et analyser l'effet — 👥 **1 pers** · 🔗 dép: N4.5

---

# 🗺️ CARTE DE PARALLÉLISATION

Vue d'ensemble de ce qui peut tourner **en même temps**. Chaque colonne = un « couloir » de travail indépendant.

```
TEMPS │
  ↓   │  COULOIR A          COULOIR B           COULOIR C            COULOIR D
──────┼───────────────────────────────────────────────────────────────────────
Jour 1│  N0.1 → N0.2        (attendre N0.1)
 début│  N0.3 / N0.4 (TOUT LE GROUPE ENSEMBLE)
──────┼───────────────────────────────────────────────────────────────────────
Jour 1│  Phase 1            Phase 2             Phase 3              Phase 5
 suite│  N1.1→N1.4          N2.1 + N2.2 (//)    N3.1→N3.7            N5.1 (trame)
      │  (1 pers)           N2.3, N2.4          (1-2 pers)           (1 pers)
      │                     (2 pers)
──────┼───────────────────────────────────────────────────────────────────────
      │         ⬇ CONVERGENCE : Phases 2 ET 3 doivent être finies ⬇
──────┼───────────────────────────────────────────────────────────────────────
Jour 2│  Phase 4 : N4.1 → N4.2 → N4.4 → N4.5   (chemin critique, 1-2 pers)
      │  En // : N4.3, N4.6
──────┼───────────────────────────────────────────────────────────────────────
Jour 2│         ⬇ Une fois N4.5 OK ⬇
 +    │  EXTENSIONS E1, E2, E3, E4 → totalement parallélisables (1 pers chacune)
      │  pendant que d'autres font E5, E6...
──────┼───────────────────────────────────────────────────────────────────────
Fin   │  Phase 5 : N5.2, N5.3 (//) → N5.5 → N5.6 → N5.7 → N5.8 (ENVOI)
```

## Règles de parallélisation à retenir

1. **Phases 1, 2, 3 et N5.1 sont 100 % indépendantes** entre elles → jusqu'à **4 couloirs en parallèle** dès le jour 1. C'est là qu'on gagne du temps avec 5 personnes.
2. **La Phase 4 est le goulot d'étranglement** 🚧 : elle a besoin que la Phase 2 (au moins Heaviside + N2.4) **et** la Phase 3 soient finies. Tant qu'elle n'est pas faite, aucune extension ne peut commencer.
3. **Les extensions E1→E4 sont parfaitement parallélisables** une fois N4.5 fait : chacun prend la sienne, aucune ne dépend d'une autre. C'est idéal pour répartir le travail à 5 en fin de projet.
4. **Ne pas mettre 3 personnes sur une tâche marquée 👥 1 pers** : elles se gêneront (conflits Git, code qui se chevauche). Mieux vaut les répartir sur des couloirs différents.
5. **Le rapport (Phase 5) se remplit en continu** : dès qu'une expérience donne un chiffre, on l'écrit. Ne pas tout garder pour la fin.

---

## 👥 Suggestion d'affectation initiale (groupe de 5)

| Personne | Couloir principal | Tâches |
|----------|-------------------|--------|
| **1 — Chef de projet / Intégration** | A puis Phase 4 | N0.1, N0.3, N4.* (assemblage), N5.7, N5.8 |
| **2 — Neurone** | B | N1.1, N2.1, N2.4 |
| **3 — Neurone (variantes + stats)** | B | N1.2, N1.3, N1.4, N2.2, N2.3, E5 |
| **4 — Données / Image** | C | N3.1 → N3.7 |
| **5 — Mesures + Rapport** | D | N5.1, N5.2, N5.3, expériences E1–E4 |

> En fin de projet, **tout le monde converge** sur les extensions et le rapport/soutenance (chacun doit pouvoir parler de sa partie en soutenance — c'est noté individuellement).
