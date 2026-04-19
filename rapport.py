"""Generate rapport.pdf with fpdf2 - EFREI Wine Quality ML Project.

Usage: python rapport.py
Requires: the notebook must have been executed first (figures/*.png must exist).
If model results are missing, this script recomputes them quickly.
"""
from __future__ import annotations

from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from fpdf import FPDF

ROOT = Path(__file__).parent
FIG = ROOT / "figures"
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
OUT = ROOT / "rapport.pdf"
RESULTS_JSON = ROOT / "results.json"


# ---------------------------------------------------------------------------
# 1. Obtenir les resultats des modeles (les recalculer si pas deja exportes)
# ---------------------------------------------------------------------------
def compute_results() -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    )
    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except ImportError:
        has_xgb = False

    RS = 42
    red = pd.read_csv(DATA / "winequality-red.csv", sep=";")
    white = pd.read_csv(DATA / "winequality-white.csv", sep=";")
    red["type"] = "red"
    white["type"] = "white"
    df = pd.concat([red, white], ignore_index=True).drop_duplicates().reset_index(drop=True)
    df["quality_binary"] = (df["quality"] >= 7).astype(int)

    y = df["quality_binary"].values
    X = df.drop(columns=["quality", "quality_binary"])
    numeric = [c for c in X.columns if c != "type"]

    prep = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first"), ["type"]),
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)

    models = {
        "Logistic Regression": Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS))]),
        "Random Forest": Pipeline([("prep", prep), ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RS, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("prep", prep), ("clf", GradientBoostingClassifier(n_estimators=300, random_state=RS))]),
    }
    if has_xgb:
        pos_w = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
        models["XGBoost"] = Pipeline([("prep", prep), ("clf", XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.1,
            scale_pos_weight=pos_w, eval_metric="logloss",
            random_state=RS, n_jobs=-1,
        ))])

    rows = []
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_proba = pipe.predict_proba(X_te)[:, 1]
        rows.append({
            "Modele": name,
            "Accuracy": round(accuracy_score(y_te, y_pred), 3),
            "Precision": round(precision_score(y_te, y_pred), 3),
            "Recall": round(recall_score(y_te, y_pred), 3),
            "F1": round(f1_score(y_te, y_pred), 3),
            "ROC-AUC": round(roc_auc_score(y_te, y_proba), 3),
        })

    out = {
        "n_total": int(len(df)),
        "n_positive": int(y.sum()),
        "positive_rate": round(float(y.mean()), 3),
        "rows": rows,
        "best": max(rows, key=lambda r: r["F1"])["Modele"],
    }
    RESULTS_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if RESULTS_JSON.exists():
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
else:
    print("Calcul des metriques des modeles en cours...")
    results = compute_results()


# ---------------------------------------------------------------------------
# 2. Classe PDF
# ---------------------------------------------------------------------------
BLUE = (0, 89, 175)    # bleu EFREI approx
DARK = (25, 25, 35)
GREY = (95, 95, 110)
LIGHT = (240, 242, 246)


class Rapport(FPDF):
    def __init__(self):
        super().__init__(format="A4", unit="mm")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)
        # Police Unicode : on cherche DejaVu dans les emplacements standards du system,
        # sinon fallback helvetica (ASCII uniquement, mais le texte est deja sans accents).
        self.base_font = "helvetica"
        dejavu_candidates = [
            ASSETS / "DejaVuSans.ttf",
            Path(r"C:/Windows/Fonts/DejaVuSans.ttf"),
        ]
        for p in dejavu_candidates:
            if p.exists():
                try:
                    self.add_font("DejaVu", "", str(p))
                    bold = p.with_name(p.stem + "-Bold" + p.suffix)
                    italic = p.with_name(p.stem + "-Oblique" + p.suffix)
                    if bold.exists():
                        self.add_font("DejaVu", "B", str(bold))
                    if italic.exists():
                        self.add_font("DejaVu", "I", str(italic))
                    self.base_font = "DejaVu"
                    break
                except Exception:
                    continue

    # -- Header / footer --------------------------------------------------
    def header(self):
        if self.page_no() == 1:
            return
        self.set_y(8)
        # Mini logo sur chaque page
        logo = ASSETS / "efrei_logo.png"
        if logo.exists():
            self.image(str(logo), x=self.w - 35, y=6, h=10)
        self.set_font(self.base_font, "B", 9)
        self.set_text_color(*BLUE)
        self.cell(0, 6, "Projet ML - Wine Quality", align="L")
        self.ln(10)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.3)
        self.line(18, self.get_y(), self.w - 18, self.get_y())
        self.ln(4)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.set_font(self.base_font, "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, f"Adam BELOUCIF - Emilien MORICE - EFREI Paris - M1 Data & IA", align="L")
        self.cell(0, 6, f"Page {self.page_no()}", align="R")

    # -- Styles -----------------------------------------------------------
    def h1(self, text):
        self.ln(4)
        self.set_font(self.base_font, "B", 18)
        self.set_text_color(*BLUE)
        self.cell(0, 11, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.set_line_width(0.6)
        self.line(18, self.get_y(), 40, self.get_y())
        self.ln(5)

    def h2(self, text):
        self.ln(3)
        self.set_font(self.base_font, "B", 13)
        self.set_text_color(*DARK)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def h3(self, text):
        self.set_font(self.base_font, "B", 11)
        self.set_text_color(*BLUE)
        self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")

    def body(self, text):
        self.set_font(self.base_font, "", 10.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.5, text)
        self.ln(1.5)

    def bullet(self, text):
        self.set_font(self.base_font, "", 10.5)
        self.set_text_color(*DARK)
        self.cell(5)
        self.cell(3, 5.5, "-")
        self.multi_cell(0, 5.5, text)
        self.ln(0.5)

    def figure(self, path: Path, caption: str, w: float = 160):
        if not path.exists():
            return
        # Centree
        x = (self.w - w) / 2
        if self.get_y() > self.h - 80:
            self.add_page()
        self.image(str(path), x=x, y=self.get_y(), w=w)
        self.ln(w * 0.72)
        self.set_font(self.base_font, "I", 9)
        self.set_text_color(*GREY)
        self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)


# ---------------------------------------------------------------------------
# 3. Construire le document
# ---------------------------------------------------------------------------
pdf = Rapport()

# ====== PAGE DE GARDE =======================================================
pdf.add_page()

# Bande bleue en haut
pdf.set_fill_color(*BLUE)
pdf.rect(0, 0, pdf.w, 35, style="F")

# Logo EFREI en grand sur fond blanc
logo = ASSETS / "efrei_logo.png"
if logo.exists():
    pdf.image(str(logo), x=(pdf.w - 90) / 2, y=55, w=90)

pdf.set_y(130)
pdf.set_font(pdf.base_font, "B", 11)
pdf.set_text_color(*BLUE)
pdf.cell(0, 8, "M1 - DATA ENGINEERING & IA", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)

pdf.set_font(pdf.base_font, "B", 26)
pdf.set_text_color(*DARK)
pdf.cell(0, 14, "Projet Machine Learning", align="C", new_x="LMARGIN", new_y="NEXT")

pdf.set_font(pdf.base_font, "", 16)
pdf.set_text_color(*GREY)
pdf.cell(0, 10, "Prediction de la qualite du vin", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "Dataset Wine Quality (UCI / Kaggle)", align="C", new_x="LMARGIN", new_y="NEXT")

pdf.ln(18)
pdf.set_draw_color(*BLUE)
pdf.set_line_width(0.5)
pdf.line(60, pdf.get_y(), pdf.w - 60, pdf.get_y())
pdf.ln(6)

pdf.set_font(pdf.base_font, "B", 13)
pdf.set_text_color(*DARK)
pdf.cell(0, 8, "Auteurs", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font(pdf.base_font, "", 12)
pdf.set_text_color(*DARK)
pdf.cell(0, 7, "Adam BELOUCIF", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Emilien MORICE", align="C", new_x="LMARGIN", new_y="NEXT")

pdf.ln(8)
pdf.set_font(pdf.base_font, "", 11)
pdf.set_text_color(*GREY)
pdf.cell(0, 6, "EFREI Paris Pantheon-Assas Universite", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 6, "Avril 2026", align="C", new_x="LMARGIN", new_y="NEXT")

# Bande bleue en bas
pdf.set_fill_color(*BLUE)
pdf.rect(0, pdf.h - 15, pdf.w, 15, style="F")


# ====== TABLE DES MATIERES ==================================================
pdf.add_page()
pdf.h1("Sommaire")
toc = [
    ("1. Probleme", 3),
    ("2. Donnees", 3),
    ("3. Analyse exploratoire (EDA)", 4),
    ("4. Preparation", 6),
    ("5. Modelisation", 7),
    ("6. Evaluation", 8),
    ("7. Analyse", 10),
    ("8. Conclusion", 11),
]
pdf.set_font(pdf.base_font, "", 12)
pdf.set_text_color(*DARK)
for title, page in toc:
    pdf.cell(0, 8, f"{title}", new_x="LMARGIN", new_y="NEXT")


# ====== 1. PROBLEME =========================================================
pdf.add_page()
pdf.h1("1. Probleme")
pdf.h2("Contexte")
pdf.body(
    "La qualite d'un vin est habituellement evaluee par des oenologues experts, "
    "ce qui est couteux, lent et subjectif. Les proprietes physico-chimiques d'un vin "
    "(acidite, sucre residuel, alcool, sulfates, etc.) sont mesurables a bas cout par "
    "des instruments de laboratoire. La question posee est donc : peut-on predire la "
    "qualite percue d'un vin a partir de ses seules caracteristiques physico-chimiques ?"
)
pdf.h2("Objectif metier")
pdf.body("Fournir a un producteur viticole un outil de scoring automatique qui :")
pdf.bullet("distingue les vins de haute qualite (note >= 7, gamme premium) des vins ordinaires (note < 7),")
pdf.bullet("identifie les variables physico-chimiques qui influencent le plus la qualite, pour piloter le processus de vinification,")
pdf.bullet("segmente la production en profils pour cibler differentes gammes de marche.")
pdf.body(
    "Pertinence : reduction du cout d'expertise, standardisation de l'evaluation, "
    "optimisation continue des process de production."
)


# ====== 2. DONNEES ==========================================================
pdf.add_page()
pdf.h1("2. Donnees")
pdf.h2("Source")
pdf.body(
    "Dataset public Wine Quality du UCI Machine Learning Repository, disponible egalement sur Kaggle "
    "sous l'identifiant uciml/red-wine-quality-cortez-et-al-2009. Deux fichiers : vins rouges "
    "et vins blancs portugais (variete Vinho Verde)."
)
pdf.body(
    "Reference : Cortez, Cerdeira, Almeida, Matos, Reis, Modeling wine preferences by data "
    "mining from physicochemical properties, Decision Support Systems 47(4), 2009."
)
pdf.h2("Description")
pdf.bullet(f"{results['n_total']} enregistrements apres deduplication (rouges + blancs fusionnes).")
pdf.bullet("11 variables explicatives numeriques : acidite fixe, acidite volatile, acide citrique, sucre residuel, chlorures, SO2 libre, SO2 total, densite, pH, sulfates, alcool.")
pdf.bullet("1 variable cible quality (score entier de 0 a 10).")
pdf.bullet("1 variable ajoutee type (red / white) pour encoder l'origine.")
pdf.bullet(f"Cible binaire derivee : quality_binary = (quality >= 7), taux positifs = {results['positive_rate']:.1%}.")

pdf.h2("Limitations")
pdf.bullet("Dataset fortement desequilibre : la majorite des vins ont une note mediane (5-6).")
pdf.bullet("Vins d'une seule region (Vinho Verde), faible generalisation geographique.")
pdf.bullet("Notes subjectives moyennes de 3 juges.")
pdf.bullet("Pas de variables contextuelles (millesime, temperature de fermentation, cepage precis).")


# ====== 3. EDA ==============================================================
pdf.add_page()
pdf.h1("3. Analyse exploratoire (EDA)")
pdf.h2("Points cles")
pdf.bullet("Aucune valeur manquante detectee, quelques doublons exacts (~1000) supprimes.")
pdf.bullet("Classes fortement desequilibrees : environ 20% seulement de vins qualifies de bons (note >= 7).")
pdf.bullet("L'alcool est la variable la plus correlee positivement avec la qualite (~ +0.44).")
pdf.bullet("Les sulfates et le type (rouge/blanc) completent le signal principal.")
pdf.bullet("L'acidite volatile est correlee negativement (-0.27) : trop d'acetate degrade le gout.")
pdf.bullet("Densite et sucre residuel sont redondants entre eux.")

pdf.figure(FIG / "01_target_distribution.png", "Figure 1 - Distribution des scores de qualite et binarisation de la cible.", w=170)
pdf.figure(FIG / "02_correlation_heatmap.png", "Figure 2 - Matrice de correlation des variables physico-chimiques.", w=160)
pdf.figure(FIG / "03_feature_distributions.png", "Figure 3 - Distribution des features selon la classe cible (rouge = ordinaire, vert = bon).", w=175)

pdf.h2("Analyse non supervisee : PCA + KMeans")
pdf.body(
    "Une analyse en composantes principales (PCA) 2D et un clustering KMeans ont ete realises "
    "sur les variables standardisees, sans utiliser la cible. Les deux premieres composantes "
    "capturent environ 50% de la variance totale."
)
pdf.figure(FIG / "04_pca_projection.png", "Figure 4 - Projection PCA des vins, coloree par type puis par classe de qualite.", w=175)
pdf.figure(FIG / "05_kmeans_elbow.png", "Figure 5 - Methode du coude : k = 3 clusters retenu.", w=130)
pdf.figure(FIG / "06_kmeans_clusters.png", "Figure 6 - Segmentation KMeans (k = 3) projetee sur le plan PCA.", w=140)

pdf.body(
    "KMeans retrouve spontanement trois profils qui correspondent grossierement a des vins "
    "blancs doux, des vins blancs secs de qualite et des vins rouges. La proportion de bons "
    "vins varie entre clusters, ce qui confirme qu'une partie du signal qualite est deja "
    "presente sans supervision."
)


# ====== 4. PREPARATION ======================================================
pdf.add_page()
pdf.h1("4. Preparation")
pdf.h2("Nettoyage")
pdf.bullet("Fusion des deux fichiers red + white avec une variable type ajoutee.")
pdf.bullet("Suppression des doublons exacts (potentielles erreurs de saisie).")
pdf.bullet("Verification des valeurs manquantes : aucune.")
pdf.bullet("Outliers conserves car ce sont de vrais vins, le scaling les absorbe.")

pdf.h2("Ingenierie de features")
pdf.bullet("Binarisation de la cible : quality_binary = (quality >= 7).")
pdf.bullet("Encodage one-hot du type de vin (red / white) avec drop de la premiere modalite.")
pdf.bullet("Standardisation des variables numeriques via StandardScaler.")
pdf.bullet("Encapsulation dans un ColumnTransformer + Pipeline sklearn : les transformations sont apprises uniquement sur le train, puis appliquees au test, ce qui evite toute fuite de donnees.")

pdf.h2("Split train/test")
pdf.body(
    "Decoupage stratifie 80% / 20% (random_state = 42). La stratification garantit que la "
    "proportion de la classe minoritaire (bons vins) est preservee entre train et test."
)


# ====== 5. MODELISATION =====================================================
pdf.add_page()
pdf.h1("5. Modelisation")
pdf.h2("Modeles implementes")
pdf.bullet("Logistic Regression - baseline lineaire interpretable.")
pdf.bullet("Random Forest - methode d'ensemble par bagging, robuste aux non-linearites.")
pdf.bullet("Gradient Boosting (sklearn) - methode d'ensemble par boosting.")
pdf.bullet("XGBoost - boosting de gradient optimise, reference sur donnees tabulaires.")
pdf.h2("Justification")
pdf.body(
    "La regression logistique sert de point de reference. Random Forest et Boosting sont "
    "explicitement requis par le cahier des charges. XGBoost apporte une optimisation "
    "supplementaire, souvent championne sur ce type de dataset."
)
pdf.body(
    "Tous les modeles sont construits via un meme Pipeline (preprocessing + classifieur) "
    "pour garantir une comparaison equitable. Les classes desequilibrees sont compensees "
    "par class_weight=balanced (LogReg, Random Forest) et scale_pos_weight (XGBoost)."
)


# ====== 6. EVALUATION =======================================================
pdf.add_page()
pdf.h1("6. Evaluation")
pdf.h2("Metriques")
pdf.bullet("Accuracy - informative mais trompeuse sur classes desequilibrees.")
pdf.bullet("Precision - parmi les vins predits bons, quel pourcentage l'est vraiment.")
pdf.bullet("Recall - parmi les vrais bons vins, combien sont retrouves.")
pdf.bullet("F1-score - moyenne harmonique precision/recall, metrique principale choisie.")
pdf.bullet("ROC-AUC - qualite globale du classement probabiliste.")

pdf.h2("Tableau comparatif")

def draw_table(pdf, headers, rows):
    col_w = [55, 24, 24, 24, 22, 24]
    pdf.set_font(pdf.base_font, "B", 10)
    pdf.set_fill_color(*BLUE)
    pdf.set_text_color(255, 255, 255)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 8, h, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_font(pdf.base_font, "", 10)
    pdf.set_text_color(*DARK)
    best = results["best"]
    for r in rows:
        is_best = (r["Modele"] == best)
        if is_best:
            pdf.set_fill_color(*LIGHT)
            pdf.set_font(pdf.base_font, "B", 10)
        else:
            pdf.set_fill_color(255, 255, 255)
            pdf.set_font(pdf.base_font, "", 10)
        pdf.cell(col_w[0], 7, str(r["Modele"]), border=1, fill=is_best)
        pdf.cell(col_w[1], 7, f"{r['Accuracy']:.3f}", border=1, align="C", fill=is_best)
        pdf.cell(col_w[2], 7, f"{r['Precision']:.3f}", border=1, align="C", fill=is_best)
        pdf.cell(col_w[3], 7, f"{r['Recall']:.3f}", border=1, align="C", fill=is_best)
        pdf.cell(col_w[4], 7, f"{r['F1']:.3f}", border=1, align="C", fill=is_best)
        pdf.cell(col_w[5], 7, f"{r['ROC-AUC']:.3f}", border=1, align="C", fill=is_best)
        pdf.ln()

draw_table(pdf, ["Modele", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], results["rows"])
pdf.ln(2)
pdf.set_font(pdf.base_font, "I", 9)
pdf.set_text_color(*GREY)
pdf.cell(0, 5, f"Ligne en surbrillance - meilleur modele selon le F1 ({results['best']}).", new_x="LMARGIN", new_y="NEXT")

pdf.figure(FIG / "07_model_comparison.png", "Figure 7 - Comparaison visuelle des metriques sur le jeu de test.", w=175)
pdf.figure(FIG / "08_confusion_matrices.png", "Figure 8 - Matrices de confusion des quatre modeles.", w=180)
pdf.figure(FIG / "09_roc_curves.png", "Figure 9 - Courbes ROC et AUC comparees.", w=150)


# ====== 7. ANALYSE ==========================================================
pdf.add_page()
pdf.h1("7. Analyse")
pdf.h2("Interpretation des resultats")
best = results["best"]
best_row = next(r for r in results["rows"] if r["Modele"] == best)
pdf.body(
    f"Le meilleur modele sur le F1-score est {best} (F1 = {best_row['F1']:.3f}, "
    f"ROC-AUC = {best_row['ROC-AUC']:.3f}). Les methodes d'ensemble surpassent systematiquement "
    "la regression logistique, ce qui indique que la frontiere entre bons vins et vins ordinaires "
    "n'est pas lineaire et exploite des interactions entre variables physico-chimiques."
)
pdf.body(
    "Les matrices de confusion montrent une precision elevee : les modeles classent "
    "rarement un vin ordinaire comme premium, ce qui est le comportement attendu en "
    "viticulture (eviter de sur-valoriser un produit qui decevrait a la degustation)."
)

pdf.figure(FIG / "10_feature_importance.png", "Figure 10 - Importance des variables selon le meilleur modele d'ensemble.", w=155)

pdf.h2("Lecture metier")
pdf.bullet("L'alcool reste la variable la plus discriminante : les bons vins ont en moyenne un taux plus eleve (signe de maturite des raisins).")
pdf.bullet("Les sulfates et l'acidite volatile sont les deux autres leviers majeurs.")
pdf.bullet("Le type de vin (rouge/blanc) ameliore legerement les performances sans etre suffisant a lui seul.")
pdf.bullet("Densite et sucre residuel apportent un signal redondant mais utile pour separer les profils.")


# ====== 8. CONCLUSION =======================================================
pdf.add_page()
pdf.h1("8. Conclusion")
pdf.h2("Recommandations")
pdf.bullet("Deployer le modele boosting retenu comme outil de pre-screening dans le laboratoire qualite : reduction massive du temps d'expertise humaine.")
pdf.bullet("Piloter la vinification en maximisant les variables a fort impact positif : viser un alcool superieur a 11%, des sulfates eleves, et limiter l'acidite volatile sous 0.4 g/L.")
pdf.bullet("Segmenter la gamme commerciale selon les clusters KMeans : chaque profil correspond a un positionnement marche distinct.")
pdf.bullet("Retenir le F1-score comme KPI de suivi plutot que l'accuracy, compte tenu du desequilibre de classes.")
pdf.h2("Limitations et travaux futurs")
pdf.bullet("Dataset geographiquement restreint (Vinho Verde) : collecter des vins d'autres regions pour generaliser le modele.")
pdf.bullet("Cible subjective (moyenne de 3 juges) : un protocole d'annotation plus large ameliorerait le signal.")
pdf.bullet("Tester des modeles supplementaires : CatBoost, LightGBM, reseaux tabulaires type TabNet.")
pdf.bullet("Tuning d'hyperparametres systematique via GridSearchCV ou Optuna.")
pdf.bullet("Passer en regression ordinale sur le score 0-10 plutot que la binarisation.")
pdf.bullet("Ajouter des features exogenes (millesime, temperature de fermentation, cepage) pour capturer plus de variance.")

pdf.ln(6)
pdf.set_draw_color(*BLUE)
pdf.set_line_width(0.3)
pdf.line(18, pdf.get_y(), pdf.w - 18, pdf.get_y())
pdf.ln(4)
pdf.set_font(pdf.base_font, "I", 10)
pdf.set_text_color(*GREY)
pdf.multi_cell(0, 6,
    "Projet realise par Adam BELOUCIF et Emilien MORICE dans le cadre du "
    "Master 1 Data Engineering & IA, EFREI Paris Pantheon-Assas Universite, "
    "annee universitaire 2025-2026."
)

pdf.output(str(OUT))
print(f"PDF genere : {OUT} ({OUT.stat().st_size // 1024} Ko)")
