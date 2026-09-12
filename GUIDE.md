# Financial Data Structures — Guide complet

Objectif de ce document : que tu puisses défendre chaque choix de ce
projet en entretien technique, pas juste dire "j'ai codé des bars".

---

## 1. Le problème que les bars alternatifs résolvent

Un "bar" OHLCV est une agrégation d'un flux continu de trades en points
discrets. La méthode par défaut de toute la finance — un bar par unité
de temps fixe (1 minute, 5 minutes...) — a un défaut structurel :

**Le marché ne produit pas de l'information à débit constant.**

- À l'ouverture, à la clôture, lors d'une news : des centaines de
  trades porteurs d'information arrivent en quelques secondes.
- À midi, en léthargie : rien ne se passe pendant de longues minutes.

Échantillonner par temps fixe revient à :
1. **Sur-échantillonner le bruit** : des bars quasi vides pendant les
   périodes calmes, qui n'apportent presque aucune information mais
   comptent autant qu'un bar actif dans un modèle en aval.
2. **Sous-échantillonner l'information** : un choc massif d'activité
   (news, ouverture) se retrouve compressé dans un seul bar — perte
   d'info, et un unique rendement énorme qui pollue les queues de
   distribution.

Conséquence directement mesurable : les rendements de time bars
s'écartent plus fortement d'une gaussienne (queues plus grasses,
autocorrélation plus marquée) que ceux d'autres méthodes
d'échantillonnage — **c'est exactement ce que confirment nos résultats
(section 6)**.

L'idée centrale : synchroniser l'échantillonnage sur une "horloge de
marché" (volume, dollar, ou déséquilibre d'ordres) plutôt que sur
l'horloge murale.

---

## 2. Les 4 bars "standards" (bars.py)

| Bar | Règle de clôture | Ce qu'il corrige vs time bars |
|---|---|---|
| **Time bars** | Toutes les N minutes | Baseline — rien |
| **Tick bars** | Toutes les N transactions | Le nombre de *trades*, pas leur taille, devient l'horloge |
| **Volume bars** | Dès que ΣV (volume physique) atteint un seuil | Corrige aussi le fait qu'un trade de 10 000 actions ≠ un trade de 10 actions |
| **Dollar bars** | Dès que Σ(price×volume) atteint un seuil | Corrige en plus la dérive du prix dans le temps : à prix doublé, le même montant économique correspond à un volume physique divisé par deux |

**Pourquoi dollar bars > volume bars sur un historique long ?**
Si une action passe de 50$ à 200$ sur plusieurs années (ou split
inversement), un seuil de volume fixe capture une quantité d'activité
*économique* très différente au début et à la fin de la période. Le
seuil en dollars, lui, reste stable — c'est pour ça que López de Prado
le recommande comme bar par défaut.

**Calibration "à nombre de bars égal" (`calibrate_threshold`)** : pour
comparer les propriétés statistiques de différents bars sans biaiser
la comparaison par la taille d'échantillon, on choisit les seuils pour
que chaque méthode produise ~le même nombre de bars sur la période
(seuil = quantité totale / nombre de bars visé).

---

## 3. Les information-driven bars (imbalance_bars.py) — la partie avancée

### 3.1 Intuition

Toute activité n'est pas porteuse d'information. Un flux de trades
équilibré (autant d'achats que de ventes) peut être très actif sans
qu'aucune direction ne se dégage. À l'inverse, un **déséquilibre
persistant** entre achats et ventes est la signature d'un agent
informé en train d'exécuter une position — c'est précisément l'instant
où on veut échantillonner un nouveau bar.

### 3.2 Formalisation (notation AFML — López de Prado, ch. 2.3)

Pour chaque trade *t*, on définit un signe `b_t ∈ {-1, +1}` :
- s'il y a un carnet d'ordres : `b_t = +1` si le trade est au ask
  (achat agressif), `-1` si au bid.
- sinon (cas le plus courant en pratique) : la **tick rule**
  (`apply_tick_rule` dans `data_acquisition.py`) :
  ```
  b_t = b_{t-1}                      si price_t == price_{t-1}
  b_t = sign(price_t - price_{t-1})  sinon
  ```

Déséquilibre cumulé depuis le début du bar courant :
```
theta_T = Σ_{t=1}^{T} b_t * v_t
```
où `v_t` est le poids du trade (1 pour tick imbalance, volume_t pour
volume imbalance, dollar_t pour **dollar imbalance** — celui qu'on
implémente).

On ferme le bar dès que `|theta_T|` dépasse un **seuil attendu** :
```
E_0[theta] ≈ E[T] * |E[b]| * E[v]
```
Ce seuil n'est **pas fixe** : il est ré-estimé en continu par moyenne
mobile exponentielle (EWMA) à partir de l'historique récent — c'est ce
qui rend le bar "intelligent" : le seuil se resserre automatiquement
quand le marché devient directionnel (bars plus courts = échantillonnage
plus fin exactement quand l'info arrive), et se desserre en marché
équilibré.

### 3.3 Un piège d'implémentation que j'ai rencontré (et corrigé) — à connaître pour l'entretien

Une première version estimait `E[b]` en faisant l'EWMA des `theta/T`
**réalisés sur les derniers bars fermés**. Résultat : instabilité
totale.

**Pourquoi ça casse** : `theta/T` sur un bar de quelques trades est un
estimateur très bruité. Comme il ré-alimente directement le seuil du
bar suivant, on crée une **boucle de rétroaction** :
- un bar se ferme "par hasard" plus tôt que la moyenne (bruit)
- → `E[T]` (EWMA des longueurs) diminue
- → le seuil du bar suivant diminue (il dépend de `E[T]`)
- → le bar suivant se ferme encore plus tôt
- → … effondrement de tous les bars suivants vers la taille minimale.

**La correction (implémentée)** : estimer `P[b=+1]` et `E[v]` par EWMA
**continue, tick par tick**, sur une fenêtre glissante de taille ~E[T]
— on lisse sur des milliers d'observations plutôt que sur une poignée
de bars, ce qui rend l'estimateur beaucoup moins bruité. On ajoute
aussi un **plancher et un plafond explicites** sur `E[T]` pour casser
la boucle de rétroaction si elle se reproduit malgré tout, et un
**garde-fou anti-runaway** (fermeture forcée au-delà d'une taille
déraisonnable) pour le cas symétrique où le seuil diverge.

C'est un vrai piège pratique, documenté dans le code
(`imbalance_bars.py`), et c'est exactement le genre de détail qui fait
la différence en entretien : tu ne redis pas juste la formule du
papier, tu sais pourquoi l'implémentation naïve échoue.

### 3.4 Run bars

Variante : au lieu de sommer tous les signes (les achats et ventes se
compensant dans `theta`), on suit séparément le cumul acheteur et le
cumul vendeur, et on ferme le bar quand le **plus grand des deux**
dépasse un seuil attendu. Un flux qui alterne achat/vente en gros
blocs (buy-buy-buy-sell-sell-sell) produit des runs longs même si
l'imbalance nette sur la fenêtre est proche de 0 — les run bars
captent donc un pattern différent des imbalance bars : l'agitation
directionnelle du carnet, pas seulement le déséquilibre net.

---

## 4. Les données : pourquoi simulées, et leurs limites

Le sandbox d'exécution n'a pas accès aux APIs de données de marché
(seuls pypi/npm/github sont autorisés en sortie réseau). Les données
tick sont donc **simulées** avec une microstructure volontairement
réaliste :

1. **Intensité d'arrivée non-homogène** : courbe en U intra-journalière
   (forte activité ouverture/clôture, creux le midi — pattern observé
   sur la quasi-totalité des marchés actions).
2. **Régime caché** (Markov 2 états, calme/stressé) qui module
   l'intensité des arrivées, la taille des volumes, et la persistance
   du signe des trades.
3. **Biais directionnel de bloc** : pendant un régime stressé, un
   biais marginal persistant (ex. 60/40) simule un agent informé qui
   accumule ou distribue une position — c'est ce qui crée un vrai
   déséquilibre net détectable par les imbalance/run bars (une simple
   autocorrélation de signe sans biais marginal ne suffit pas : elle
   se compense en moyenne et ne produit rien de systématique à
   détecter).
4. **Impact prix avec retour à la moyenne** : chaque trade signé pousse
   le prix dans sa direction, avec une force de rappel vers le prix
   d'ouverture du jour (sans quoi un biais de bloc, cumulé sur des
   milliers de trades, fait diverger le prix exponentiellement — bug
   rencontré et corrigé pendant le développement).

**Limite à assumer en entretien** : ce ne sont *pas* de vraies données.
Le pipeline (bars.py, imbalance_bars.py, statistical_tests.py) est
entièrement agnostique à la source — pour brancher du vrai tick data
(Databento, LOBSTER, Kaggle, export MT5/broker), il suffit de
remplacer `generate_synthetic_ticks()` par un loader qui renvoie un
DataFrame au même format (`price`, `volume`, `sign`, `dollar`, indexé
par timestamp). Mentionne-le proactivement à l'oral : ça montre que tu
comprends la différence entre "prouver le concept" et "l'avoir
validé sur données réelles".

---

## 5. Les 4 tests statistiques (statistical_tests.py) — et ce qu'ils testent VRAIMENT

| Test | H0 | Ce qu'on regarde |
|---|---|---|
| **Jarque-Bera** | Les rendements sont gaussiens | La *statistique* (pas juste rejet/non-rejet — **tous** les bars rejettent la normalité, l'intérêt est l'amplitude du rejet) |
| **Ljung-Box** | Pas d'autocorrélation des rendements bruts | p-value au lag 1 surtout — une autocorrélation significative est souvent le signe d'un échantillonnage inadapté (bid-ask bounce, etc.) |
| **ARCH-LM** | Pas d'hétéroscédasticité conditionnelle (volatility clustering) | Testé sur les rendements au carré ; presque tous les bars montrent un effet ARCH (fait stylisé quasi-universel), on compare l'amplitude |
| **ADF** | Racine unitaire (non-stationnarité) | Garde-fou : si les rendements ne sont pas stationnaires, comparer les tests ci-dessus n'a pas de sens |

**Point d'attention pour l'entretien** : ne jamais dire "le test montre
que les rendements sont normaux". Aucune série de rendements financiers
réels n'est gaussienne — le test JB est **toujours** rejeté (p-value ~0).
Ce qu'on compare, c'est **l'amplitude** de la statistique JB et de
l'excess kurtosis entre types de bars : plus c'est bas, plus la
distribution se rapproche d'une gaussienne, même si elle ne l'atteint
jamais.

---

## 6. Résultats obtenus (voir RESULTS.md et les graphiques)

Sur nos 261 323 trades simulés / 5 jours :

- **Dollar bars** : statistique JB la plus faible (~1.5), kurtosis
  quasi nulle (-0.08) → la distribution la plus proche d'une
  gaussienne parmi les 6 méthodes. Cohérent avec la littérature
  (López de Prado recommande dollar bars comme choix par défaut).
- **Time bars** : statistique JB de loin la plus élevée (~447),
  kurtosis très positive (5.2), autocorrélation lag-1 la plus forte en
  valeur absolue (-0.32) → la méthode la moins bien comportée
  statistiquement, exactement comme attendu.
- **Tick et volume bars** : intermédiaires, nettement meilleurs que
  time bars mais moins bons que dollar bars.
- **Dollar imbalance bars** : résultat contre-intuitif — kurtosis très
  élevée (12.3) et JB très haute. **Explication** : la distribution de
  la taille des bars est extrêmement hétérogène (voir
  `bar_size_distribution.png` — médiane ~13 trades, mais quelques bars
  de plusieurs centaines de trades). Beaucoup de bars minuscules
  produisent des rendements quasi nuls (pic très pointu au centre de
  l'histogramme), tandis qu'une minorité de bars capturent un vrai
  choc directionnel (queues grasses). C'est un vrai phénomène décrit
  dans la littérature (les imbalance bars ont une variance de taille
  intrinsèquement plus élevée que les autres bars), mais ça montre
  aussi une limite pratique : sans filtrage des bars trop courts
  (`min_ticks` plus agressif), l'imbalance bar peut être *dominé* par
  du bruit de microstructure plutôt que par la vraie information.
- **Run bars** : comportement intermédiaire, plus proche des tick/volume
  bars que des dollar imbalance bars — cohérent avec une taille de bar
  beaucoup moins dispersée (voir boxplot).

**Ce que tu dois pouvoir dire en entretien** : "Sur mes données, dollar
bars donne les meilleures propriétés statistiques, ce qui confirme la
littérature. Les imbalance bars nécessitent un tuning plus fin (seuil
minimum de longueur, span de l'EWMA) pour ne pas être dominées par du
bruit de microstructure sur des bars trop courts — c'est une limite
connue de l'approche naïve que j'ai documentée et partiellement
corrigée."

---

## 7. Pour aller plus loin (extensions possibles)

1. **Tick imbalance bars et volume imbalance bars** : mêmes formules
   que dollar imbalance bars, juste `v_t = 1` ou `v_t = volume_t` —
   trivial à ajouter dans `imbalance_bars.py` en généralisant la
   fonction avec un paramètre `weight_col`.
2. **Filtrer les bars trop courts** : augmenter `min_ticks` (fixé à 10
   ici) et étudier l'effet sur JB/kurtosis pour les imbalance/run bars
   — devrait réduire l'artefact observé en section 6.
3. **Tester sur plusieurs séries / plusieurs jours indépendamment**
   pour vérifier la robustesse des résultats (nos 5 jours partagent la
   même dynamique de régime — un vrai test demanderait plus de
   diversité de conditions de marché).
4. **Brancher du vrai tick data** (Databento offre des échantillons
   gratuits, LOBSTER pour le LOB NASDAQ, ou export MT5 en mode tick)
   et refaire tourner exactement le même pipeline sans rien changer
   d'autre que `data_acquisition.py`.
5. **CUSUM filter sur les bars obtenus** (AFML ch. 2.5) : une étape
   naturelle après la construction des bars, pour détecter les
   événements significatifs et construire le "triple-barrier labeling"
   utilisé ensuite pour du ML supervisé — pont direct vers ton projet
   de dashboard trading / génération d'alphas.

---

## 8. Questions pour t'auto-tester (mastery check)

Essaie d'y répondre sans relire le code — si tu bloques, c'est le
signal pour retourner sur cette section précise :

1. Pourquoi un time bar de 5 minutes est-il statistiquement moins bien
   comporté qu'un dollar bar, en une phrase ?
2. Pourquoi dollar bars est-il préféré à volume bars sur un historique
   long (plusieurs années) ?
3. Qu'est-ce que la tick rule, et quand en a-t-on besoin ?
4. Pourquoi une estimation de `E[b]` par EWMA sur les *bars fermés*
   est-elle instable, et qu'est-ce qui la stabilise ?
5. Quelle est la différence conceptuelle entre un imbalance bar et un
   run bar ?
6. Le test de Jarque-Bera rejette la normalité pour TOUS les types de
   bars dans nos résultats — est-ce que ça veut dire que le test ne
   sert à rien ? Pourquoi ?
7. Pourquoi calibre-t-on les seuils pour obtenir un nombre de bars
   comparable entre méthodes avant de comparer leurs propriétés
   statistiques ?
