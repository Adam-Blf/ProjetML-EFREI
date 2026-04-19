<div align="center">

# 🍷 Projet Machine Learning · Wine Quality

*Prédiction de la qualité d'un vin à partir de ses propriétés physico-chimiques*

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-006FB9?style=flat)](https://xgboost.ai/)
[![Pandas](https://img.shields.io/badge/pandas-2.2-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-11557C?style=flat)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat)](LICENSE)
[![EFREI](https://img.shields.io/badge/EFREI-Paris_Panth%C3%A9on--Assas-0059AF?style=flat)](https://www.efrei.fr/)
[![M1](https://img.shields.io/badge/M1-Data_Engineering_%26_IA-6366F1?style=flat)](https://www.efrei.fr/programmes-experts/mastere-data-engineering/)

</div>

---

## 👥 Auteurs

| | |
|---|---|
| **Adam BELOUCIF** | [![GitHub](https://img.shields.io/badge/GitHub-Adam--Blf-181717?style=flat&logo=github&logoColor=white)](https://github.com/Adam-Blf) [![Email](https://img.shields.io/badge/Email-adambeloucif%40gmail.com-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:adambeloucif@gmail.com) |
| **Emilien MORICE** | *Binôme M1 Data Engineering & IA · EFREI Paris* |

---

## 🎯 Contexte

Projet académique du **Mastère Data Engineering & IA** à l'EFREI Paris Panthéon-Assas · Master 1 · avril 2026.

La qualité d'un vin est habituellement évaluée par des œnologues experts, ce qui est coûteux et subjectif. Les propriétés physico-chimiques d'un vin sont mesurables à bas coût. **Objectif** : automatiser le scoring de qualité à partir de ces mesures.

## 🧪 Méthodologie

### Données
- **Dataset** : [Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality) (UCI / Kaggle) — Vinho Verde portugais
- **Taille** : ~5 300 vins (rouges + blancs) après déduplication
- **Cible** : classification binaire `quality >= 7` (bon vin vs ordinaire) · ~20 % de positifs

### Pipeline
- `ColumnTransformer` + `StandardScaler` + `OneHotEncoder` (zéro fuite train/test)
- Encapsulation de chaque modèle dans un `sklearn.pipeline.Pipeline`
- Split stratifié 80 / 20 · cross-validation 5-folds

### Modèles comparés
| Modèle | Famille |
|---|---|
| Logistic Regression | Baseline linéaire |
| **Random Forest** | Ensemble — bagging |
| **Gradient Boosting** | Ensemble — boosting (sklearn) |
| **XGBoost** | Ensemble — boosting optimisé |

### Analyse non supervisée
- **PCA** 2D — réduction de dimension pour visualisation
- **KMeans** (k = 3) — segmentation en profils de vins

## 📊 Résultats

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.731 | 0.395 | 0.782 | 0.525 | 0.817 |
| Random Forest | 0.846 | 0.698 | 0.332 | 0.450 | 0.878 |
| Gradient Boosting | 0.844 | 0.648 | 0.391 | 0.488 | 0.852 |
| **XGBoost** ⭐ | **0.834** | **0.557** | **0.609** | **0.582** | **0.858** |

**Meilleur modèle (F1)** : XGBoost · il équilibre le mieux précision et recall sur la classe minoritaire.

**Variables les plus discriminantes** : alcool · sulfates · acidité volatile.

## 📦 Contenu du dépôt

```
ProjetML-EFREI/
├── notebook.ipynb       Notebook Jupyter exécuté (8 sections imposées)
├── rapport.pdf          Rapport final livré (mise en page EFREI)
├── data/                Dataset Wine Quality (rouges + blancs)
├── README.md            Ce fichier
├── requirements.txt     Dépendances Python
└── LICENSE              MIT
```

## 🚀 Reproduction

```bash
git clone https://github.com/Adam-Blf/ProjetML-EFREI.git
cd ProjetML-EFREI
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## 📚 Structure du notebook (cahier des charges)

1. **Problème** · contexte métier + objectifs
2. **Données** · source, description, limitations
3. **EDA** · analyse exploratoire + PCA + KMeans
4. **Préparation** · nettoyage + feature engineering
5. **Modélisation** · 4 modèles comparés
6. **Évaluation** · métriques adaptées au déséquilibre
7. **Analyse** · interprétation + importance des variables
8. **Conclusion** · recommandations + travaux futurs

## 📄 Licence

Code sous licence [MIT](LICENSE). Dataset Wine Quality © Cortez et al., 2009.

---

<div align="center">

*Projet réalisé dans le cadre du [Mastère Data Engineering & IA](https://www.efrei.fr/programmes-experts/mastere-data-engineering/) · [EFREI Paris Panthéon-Assas](https://www.efrei.fr) · 2025-2026*

</div>
