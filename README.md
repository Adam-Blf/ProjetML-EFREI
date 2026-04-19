# Projet Machine Learning - Wine Quality

> EFREI Paris Pantheon-Assas Universite - M1 Data Engineering & IA - Avril 2026

Projet de Machine Learning realise par **Adam BELOUCIF** et **Emilien MORICE** sur le dataset *Wine Quality* (UCI / Kaggle). L'objectif : predire si un vin est de qualite premium (note >= 7) a partir de ses seules proprietes physico-chimiques.

## Sommaire

1. [Contexte et objectifs](#contexte-et-objectifs)
2. [Dataset](#dataset)
3. [Pipeline et modeles](#pipeline-et-modeles)
4. [Resultats](#resultats)
5. [Structure du repo](#structure-du-repo)
6. [Installation](#installation)
7. [Utilisation](#utilisation)
8. [Livrables](#livrables)
9. [Auteurs](#auteurs)

## Contexte et objectifs

Contexte metier : l'evaluation de la qualite d'un vin par un oenologue est couteuse et subjective. A l'inverse, ses proprietes physico-chimiques sont mesurees a bas cout. Peut-on automatiser le scoring ?

Objectifs :

- **Classification binaire** : distinguer les bons vins (quality >= 7) des vins ordinaires.
- **Interpretation** : identifier les variables physico-chimiques les plus discriminantes.
- **Segmentation** : decouvrir des profils de vins par apprentissage non supervise.

## Dataset

- **Source** : UCI Machine Learning Repository, egalement disponible sur Kaggle (`uciml/red-wine-quality-cortez-et-al-2009`).
- **Taille** : ~6 500 vins rouges et blancs portugais (Vinho Verde) apres fusion, ~5 300 apres deduplication.
- **Variables** : 11 features physico-chimiques + 1 cible `quality` (0-10) + variable ajoutee `type` (red/white).
- **Desequilibre** : ~20% seulement de vins classes "bons" (>= 7).

## Pipeline et modeles

Pipeline construit avec `sklearn.pipeline.Pipeline` + `ColumnTransformer` pour garantir zero fuite de donnees entre train et test :

1. **Nettoyage** : deduplication, verification missing values.
2. **Feature engineering** : binarisation de la cible, OneHot sur `type`, StandardScaler sur numeriques.
3. **EDA non supervisee** : PCA 2D + KMeans pour visualiser la structure latente.
4. **Modeles compares** :
   - Logistic Regression (baseline)
   - **Random Forest** (ensemble bagging)
   - **Gradient Boosting** (sklearn)
   - **XGBoost** (boosting optimise)
5. **Evaluation** : cross-validation 5-folds stratifies, metriques Accuracy / Precision / Recall / F1 / ROC-AUC.

## Resultats

Le meilleur modele selon le F1-score est un modele d'ensemble (Random Forest ou XGBoost selon l'execution). Les scores typiques obtenus :

| Modele              | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|:--------:|:---------:|:------:|:-----:|:-------:|
| Logistic Regression |   ~0.74  |    ~0.43  | ~0.73  | ~0.54 |  ~0.80  |
| Random Forest       |   ~0.88  |    ~0.73  | ~0.56  | ~0.64 |  ~0.90  |
| Gradient Boosting   |   ~0.87  |    ~0.73  | ~0.54  | ~0.62 |  ~0.90  |
| XGBoost             |   ~0.86  |    ~0.62  | ~0.70  | ~0.66 |  ~0.90  |

Les valeurs exactes sont dans `results.json` genere apres execution.

**Features les plus importantes** : alcool, sulfates, acidite volatile.

## Structure du repo

```
ProjetML-EFREI/
├── data/                       Dataset UCI/Kaggle (red + white)
├── figures/                    Graphiques generes par le notebook
├── assets/                     Logo EFREI + polices
├── prof/                       Documents d'enonce fournis
├── notebook.ipynb              Notebook Jupyter (8 sections demandees)
├── build_notebook.py           Script de regeneration du notebook
├── rapport.py                  Generation du PDF final (fpdf2)
├── rapport.pdf                 Rapport livre
├── results.json                Metriques exportees
├── requirements.txt            Dependances Python
└── README.md
```

## Installation

Prerequis : Python 3.10+.

```bash
git clone https://github.com/Adam-Blf/ProjetML-EFREI.git
cd ProjetML-EFREI
pip install -r requirements.txt
```

## Utilisation

```bash
# 1. Regenerer (si besoin) et executer le notebook
python build_notebook.py
jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb

# 2. Generer le rapport PDF
python rapport.py

# 3. Ouvrir le rapport
start rapport.pdf       # Windows
open rapport.pdf        # macOS
xdg-open rapport.pdf    # Linux
```

## Livrables

- `notebook.ipynb` : etude complete, 8 sections conformes au cahier des charges (Probleme / Donnees / EDA / Preparation / Modelisation / Evaluation / Analyse / Conclusion).
- `rapport.pdf` : rapport mis en forme (page de garde EFREI, sommaire, figures embarquees, tableau comparatif).
- `figures/` : toutes les visualisations au format PNG haute resolution.

## Auteurs

- **Adam BELOUCIF** - [adambeloucif@gmail.com](mailto:adambeloucif@gmail.com)
- **Emilien MORICE**

EFREI Paris Pantheon-Assas Universite - Master 1 Data Engineering & IA - Annee 2025-2026.

## Licence

Code publie sous licence MIT (voir [LICENSE](LICENSE)). Le dataset Wine Quality est la propriete de ses auteurs (Cortez et al., 2009).
