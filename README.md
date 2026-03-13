# Drug–Target Interaction Prediction with Classical Machine Learning

## Problema
Acest proiect abordează problema predicției interacțiunilor dintre medicamente și ținte proteice (Drug–Target Interaction, DTI) folosind metode clasice de machine learning. Scopul nu a fost doar obținerea unor performanțe bune, ci și evaluarea realistă a generalizării, comparând un split random cu un split la nivel de compus, pentru a evidenția efectul de leakage asupra rezultatelor.

## Dataset
- Sursă: Kaggle — [Davis and KIBA Datasets](https://www.kaggle.com/datasets/rajaryan2315/davis-and-kiba-datasets)
- Dimensiune: 118254 exemple, 1050 features finale
- Ce conține: pentru fiecare pereche drug–target, datasetul conține `compound_iso_smiles`, `target_sequence` și `affinity`; eticheta binară a fost definită astfel: activ dacă `affinity >= 12`, inactiv altfel

## Ce am făcut
1. Am explorat datele și am analizat distribuția claselor
2. Am pregătit datele și am construit features pentru compuși și proteine
3. Am antrenat 2 familii de modele: Logistic Regression și XGBoost
4. Am evaluat modelele atât cu random split, cât și cu drug-level split
5. Am analizat screening metrics și feature importance pentru modelul final

## Ce am învățat
Prin acest proiect am înțeles mai bine cât de mult poate influența strategia de evaluare performanța aparentă a unui model de DTI. Cea mai importantă lecție a fost că random split poate supraestima semnificativ performanța, iar un drug-level split oferă o estimare mult mai realistă a generalizării. De asemenea, am învățat să construiesc un pipeline complet, de la feature engineering și antrenare până la evaluare orientată spre virtual screening și interpretarea importanței feature-urilor.

## Rezultate
| Model | ROC-AUC | PR-AUC |
|-------|--------:|-------:|
| Logistic Regression (Random Split) | 0.805 | 0.574 |
| Logistic Regression (Drug Split) | 0.659 | 0.358 |
| XGBoost (Random Split) | 0.908 | 0.758 |
| XGBoost (Drug Split) | 0.865 | 0.667 |

**Modelul final:** XGBoost evaluat cu drug-level split, cu ROC-AUC de 0.865 și PR-AUC de 0.667

## Cum să rulați
```bash
pip install -r requirements.txt

python src/models/train_baseline.py
python src/models/train_drug_split.py
python src/models/train_xgb_random_split.py
python src/models/train_xgb_drug_split.py

python src/models/plot_model_comparison.py
python src/models/summarize_metrics.py
python src/models/screening_metrics.py
python src/models/analyze_feature_importance.py
