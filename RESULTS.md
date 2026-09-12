# Résultats — Financial Data Structures (bars alternatifs)

Données : 261,323 trades simulés sur 5 jours.

## Nombre de bars et taille

| Type de bar | Nb bars | Taille moy. (trades) | Std taille |
|---|---|---|---|
| time | 390 | 670.1 | 645.0 |
| tick | 391 | 668.3 | 32.7 |
| volume | 390 | 670.1 | 290.3 |
| dollar | 391 | 668.3 | 321.5 |
| dollar_imbalance | 5159 | 50.7 | 76.1 |
| run | 472 | 553.7 | 54.7 |

## Tests statistiques sur les rendements

| Type de bar | JB stat | Skewness | Excess kurtosis | LB p-val (lag1) | ARCH p-val | ADF p-val |
|---|---|---|---|---|---|---|
| time | 447.3 | -0.372 | 5.200 | 0.0000 | 0.0000 | 0.0000 |
| tick | 10.4 | -0.039 | 0.797 | 0.0001 | 0.0016 | 0.0000 |
| volume | 8.2 | -0.154 | 0.640 | 0.0001 | 0.4885 | 0.0000 |
| dollar | 1.5 | 0.147 | -0.081 | 0.0000 | 0.0727 | 0.0000 |
| dollar_imbalance | 32625.7 | 0.169 | 12.316 | 0.0000 | 0.0000 | 0.0000 |
| run | 25.4 | 0.049 | 1.134 | 0.0000 | 0.0011 | 0.0000 |

**Bar le plus proche d'une distribution gaussienne (JB stat la plus faible) : `dollar`**
**Bar le plus éloigné (JB stat la plus élevée) : `dollar_imbalance`**
