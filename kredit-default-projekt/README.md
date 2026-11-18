# Kreditausfall-Modellierung (LendingClub)

*English summary: predicting loan default on a 500-loan LendingClub sample; comparison of
logistic regression, random forest and gradient boosting with three ways of handling class
imbalance; evaluation, feature importance and a written report (German).*

Vorhersage, ob ein Konsumentenkredit ausfällt (`default` = 1: notleidend / Charge-Off,
0 = vollständig zurückgezahlt). Schwerpunkt ist der Umgang mit dem Klassenungleichgewicht
(rund 20 % Ausfälle) und eine ehrliche Bewertung der Modelle.

## Daten

- **Quelle:** öffentlicher LendingClub-Datensatz (Peer-to-Peer-Konsumentenkredite, USA),
  über die Hugging-Face-Datasets-API (`RPD123-byte/credit-risk-datasets`,
  `lending_club_clean.csv`, 150.000 Zeilen).
- **Stichprobe:** 500 Zeilen, als 5 Blöcke à 100 Zeilen über den gesamten Indexbereich
  gezogen (`data/raw_batches/`), zusammengeführt in `data/lending_club_sample.csv`. Alle
  Werte sind echte, unveränderte Beobachtungen.
- **19 Merkmale:** Kredithöhe, Laufzeit, Zinssatz, Rate, Rating und Subrating,
  Beschäftigungsdauer, Wohnverhältnis, Einkommen, Verifizierungsstatus, Verwendungszweck,
  DTI, Zahlungsverzüge, Kreditanfragen, offene und gesamte Kreditlinien, negative Einträge,
  revolvierender Saldo und dessen Auslastung.
- **Zielverteilung:** 398 Nicht-Ausfälle, 102 Ausfälle (20,4 %); `emp_length` fehlt in
  5 % der Fälle.

## Ablauf

Alle Skripte aus dem Projektordner starten (sie nutzen relative Pfade):

```bash
cd kredit-default-projekt
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python src/01_eda.py                 # Explorative Analyse  -> results_eda.json, figures/01-04
python src/02_preprocessing.py       # Imputation, Split    -> data/X_*.csv, y_*.csv
python src/03_modeling.py            # Training & Vergleich -> models/, results_model_comparison.*
python src/04_evaluation.py          # ROC, PR, Kalibrierung, Konfusionsmatrix -> figures/05-08
python src/05_feature_importance.py  # Impurity & Permutation Importance -> figures/09-10
```

| Schritt | Inhalt |
|---|---|
| `01_eda.py` | Klassenverteilung, Kredithöhe und Einkommen je Klasse, Korrelationsmatrix, Zahlungshistorie |
| `02_preprocessing.py` | Fehlende Beschäftigungsdauer → Median + Missing-Flag; stratifizierter Split 350 / 150 |
| `03_modeling.py` | Logistische Regression (Baseline), Random Forest, Gradient Boosting mit (a) Klassengewichten, (b) SMOTE (eigene k-NN-Implementierung), (c) Random Undersampling; alle auf demselben, unveränderten Testset bewertet |
| `04_evaluation.py` | ROC- und Precision-Recall-Kurven aller Modelle, Kalibrierungskurve und Konfusionsmatrix des besten Modells |
| `05_feature_importance.py` | Impurity- und Permutation-Importance für den Random Forest (class_weight) |

## Ergebnisse (Testset, 150 Kredite, davon 31 Ausfälle)

| Modell | ROC-AUC | PR-AUC | Precision (Ausfall) | Recall (Ausfall) | F1 | Brier |
|---|---|---|---|---|---|---|
| Random Forest (SMOTE) | **0,676** | **0,313** | 0,238 | 0,161 | 0,192 | **0,173** |
| Logistische Regression (Baseline) | 0,645 | 0,289 | 0,316 | **0,581** | **0,409** | 0,227 |
| Gradient Boosting (SMOTE) | 0,604 | 0,291 | 0,235 | 0,129 | 0,167 | 0,189 |
| Random Forest (Undersampling) | 0,600 | 0,281 | **0,320** | 0,516 | 0,395 | 0,238 |
| Random Forest (class_weight) | 0,570 | 0,238 | 0,143 | 0,065 | 0,089 | 0,187 |
| Gradient Boosting (class_weight) | 0,548 | 0,248 | 0,217 | 0,161 | 0,185 | 0,211 |

**Einordnung:**
- **Bestes Modell nach ROC-AUC:** der Random Forest mit SMOTE. Mit der Standardschwelle
  erkennt er aber nur 5 der 31 Ausfälle (Konfusionsmatrix [[103, 16], [26, 5]]).
- **Bei hoher Erkennungsrate gewünscht:** die logistische Regression findet 58 % der
  Ausfälle und hat den besten F1-Wert. Sie ist die bessere Wahl, wenn möglichst viele
  Ausfälle erkannt werden sollen; ansonsten gibt es noch die Option, beim Random Forest
  die Entscheidungsschwelle zu senken.
- **Wichtigste Merkmale:** Zinssatz, Anzahl offener und gesamter Kreditlinien,
  LendingClub-Subrating und Jahreseinkommen (Permutation Importance).
- **Vorsicht bei der Interpretation:** 500 Kredite (150 im Test) sind wenig, die
  Unterschiede zwischen den Modellen liegen teils im Bereich des Zufalls. Für belastbare
  Aussagen den vollständigen Datensatz verwenden.

## Ordnerstruktur

```
kredit-default-projekt/
├── data/            Stichprobe, Rohblöcke, Train/Test-Dateien
├── src/             Skripte 01–05
├── models/          gespeicherte Modelle (.joblib) und Scaler
├── figures/         Abbildungen 01–10
├── reports/         Projektbericht (.docx, .pdf) und build_report.js
└── results_*.json/.csv   Kennzahlen der einzelnen Schritte
```

`reports/build_report.js` erzeugt den Word-Bericht aus den `results_*.json` und den
Abbildungen (Node.js mit dem Paket `docx`): `cd reports && npm install docx && node build_report.js`.

**Hinweise:**
- SHAP-Werte wurden nicht berechnet (das Paket war in der Entwicklungsumgebung nicht
  installierbar); stattdessen dient Permutation Importance als modellagnostische Analyse.
- Die Datei `.~lock.Kreditrisikomodellierung_Projektbericht.docx#` ist eine
  LibreOffice-Sperrdatei und kann gelöscht werden.
