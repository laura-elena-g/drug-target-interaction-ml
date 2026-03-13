## 1. Problem formulation

Drug–target interaction (DTI) prediction is an important task in computational drug discovery because it helps prioritize which compounds are more likely to interact with which biological targets before costly experimental validation. In practical virtual screening workflows, the goal is not only to classify interactions correctly in a general sense, but also to rank candidate compound–target pairs so that truly active pairs appear near the top of the list.

In this project, the DTI problem was formulated as a binary classification task based on the KIBA dataset. Each sample corresponds to a drug–target pair described by a compound structure, a protein sequence, and an affinity score. To obtain binary labels, pairs with affinity values greater than or equal to 12 were defined as active, while pairs with affinity values below 12 were defined as inactive. This threshold transforms a continuous binding-related signal into a classification setting suitable for classical machine learning baselines.

The project was designed with two main objectives. The first objective was to build and evaluate strong classical machine learning baselines for DTI prediction using handcrafted drug and protein features. The second objective was to examine how evaluation protocol affects the apparent performance of DTI models. In particular, the project focuses on the difference between standard random splits and drug-level splits, where compounds present in the training set are excluded from the test set. This distinction is important because random splits can produce overly optimistic estimates when the model is tested on compounds it has effectively already seen during training.

Therefore, the project is not intended only to answer whether a model can achieve good classification metrics, but also whether those metrics remain meaningful under a more realistic generalization setting. From a drug discovery perspective, the most relevant question is whether the model can prioritize active interactions for previously unseen compounds. For this reason, the study evaluates both conventional classification performance and screening-oriented behavior.

## 2. Dataset description

This project uses the KIBA dataset, a widely used benchmark for drug–target interaction prediction. The processed version used here contains 118,254 drug–target pairs, with each row describing a compound structure, a protein target sequence, and an affinity-related score. The main columns retained for modeling are `compound_iso_smiles`, `target_sequence`, and `affinity`.

The `compound_iso_smiles` column provides a text representation of the molecular structure of each compound, while `target_sequence` contains the amino acid sequence of the corresponding protein target. The `affinity` column represents the interaction strength and is used here to derive binary activity labels. To convert the problem into a binary classification task, drug–target pairs with affinity values greater than or equal to 12 were labeled as active, and pairs with lower values were labeled as inactive.

After binarization, the dataset is moderately imbalanced. Approximately 21.2% of the samples belong to the active class, while about 78.8% belong to the inactive class. This class imbalance is important because it affects the interpretation of performance metrics. In particular, metrics such as precision–recall AUC and early enrichment are more informative than accuracy alone in this setting.

The dataset was intentionally evaluated under two different splitting strategies. The first is a standard random split, in which samples are randomly divided into training and test sets. The second is a drug-level split, in which unique compounds are separated before assigning samples to training and test sets, ensuring that no drug appears in both partitions. This second setting is more challenging, but it better reflects the practical problem of generalizing to previously unseen compounds.

The inclusion of both split strategies is central to the purpose of the project. Rather than treating the dataset only as a benchmark for maximizing predictive performance, the project uses it to study how evaluation design changes the apparent quality of a DTI model. This is especially relevant in cheminformatics, where random splits can hide memorization effects and produce inflated estimates of generalization.

## 3. Feature engineering

The feature engineering pipeline was designed to generate compact and interpretable representations of both drugs and protein targets using handcrafted descriptors.

Drug features were generated from the SMILES strings with RDKit. Each compound was represented using a Morgan fingerprint with radius 2 and 1024 bits. To this fingerprint, five molecular descriptors were added: molecular weight, LogP, number of hydrogen bond donors, number of hydrogen bond acceptors, and topological polar surface area. This resulted in a 1029-dimensional drug feature vector.

Protein features were intentionally kept simple. Each target sequence was encoded using amino acid composition, which produces a 20-dimensional vector corresponding to the relative frequency of each standard amino acid. One additional feature representing sequence length was added, giving a total of 21 protein features.

The final feature matrix was obtained by concatenating the drug and protein features for each drug–target pair. This produced an input matrix of shape `(118254, 1050)`, with the first 1029 columns corresponding to drug features and the last 21 columns corresponding to protein features.

This feature design supports the main purpose of the project: building transparent and reproducible classical machine learning baselines without relying on graph neural networks, transformers, or heavy pretrained embeddings.

## 4. Modeling approach

The project was designed to compare simple linear modeling with a stronger nonlinear baseline under both optimistic and more realistic evaluation settings.

Two model families were used. The first was logistic regression, which served as a classical baseline. Because the combined feature matrix includes descriptors with different numerical scales, standardization was applied before training logistic regression models. Class imbalance was handled using balanced class weights.

The second model family was XGBoost, which was used as a stronger nonlinear baseline. The XGBoost models were trained with 300 estimators, maximum depth 6, learning rate 0.1, subsample 0.8, and column subsampling 0.8. To account for class imbalance, the positive class weight was adjusted using the ratio between inactive and active samples in the training data.

Each model was evaluated under two splitting strategies. In the random split setting, samples were divided using a stratified 80/20 train–test split. In the drug-level split setting, unique SMILES strings were first divided into training and test groups, and all pairs associated with a given compound were kept in only one partition. This ensured that no drug appeared in both training and test data.

Performance was evaluated primarily using ROC-AUC and PR-AUC. In addition, confusion matrices at a threshold of 0.5 were computed to summarize classification behavior. Because the project is motivated by virtual screening, screening-oriented metrics such as Precision@K, Recall@K, and enrichment factor were also computed later for the ranked predictions.

## 5. Results

The results show clear differences both between model families and between evaluation strategies.

For logistic regression, the random split produced a ROC-AUC of 0.805 and a PR-AUC of 0.574. Under the drug-level split, performance dropped to a ROC-AUC of 0.659 and a PR-AUC of 0.358. This indicates that the linear baseline is strongly affected by the removal of compound overlap between training and test data.

XGBoost outperformed logistic regression in both settings. Under the random split, XGBoost achieved a ROC-AUC of 0.908 and a PR-AUC of 0.758. Under the drug-level split, it still retained strong performance, with a ROC-AUC of 0.865 and a PR-AUC of 0.667. These results show that nonlinear modeling captures signal that the linear baseline misses.

A consistent pattern across both model families is that random splitting leads to better results than drug-level splitting. This supports the idea that random splits can produce optimistic estimates in DTI prediction, because compounds seen during training may also appear in the test set. In contrast, the drug-level split provides a more realistic estimate of generalization to unseen molecules.

The ROC and precision–recall comparison plots reinforce these conclusions. XGBoost produced the strongest curves overall, while the drug-level split curves were consistently lower than their random-split counterparts. Because the dataset is imbalanced, the precision–recall results are especially important and confirm that the realistic evaluation setting is substantially more challenging.

Overall, the strongest raw performance was obtained with XGBoost under a random split, but the most meaningful result for practical use is the XGBoost drug-level split model, since it reflects performance on previously unseen compounds.

## 6. Screening interpretation

Because the project is motivated by virtual screening, overall classification metrics were complemented with ranking-based screening metrics. These metrics evaluate whether active drug–target pairs are concentrated near the top of the ranked prediction list, which is more relevant than global performance alone in practical prioritization settings.

The strongest realistic screening performance was obtained with XGBoost under the drug-level split. For this model, Precision@5% was 0.866, meaning that about 86.6% of the top 5% highest-ranked predictions were active. The corresponding enrichment factor at 5% was 4.12, indicating that this top-ranked subset was more than four times richer in active pairs than random selection.

At larger cutoffs, the same pattern was maintained. For XGBoost under the drug-level split, Precision@10% was 0.771 and Precision@20% was 0.637, with enrichment factors of 3.67 and 3.03, respectively. These values show that even under the more realistic unseen-drug setting, the model remains effective at concentrating actives near the top of the ranking.

Logistic regression showed weaker screening behavior, especially under the drug-level split, where Precision@5% was 0.474 and EF@5% was 2.25. This confirms that the nonlinear model is not only better in terms of ROC-AUC and PR-AUC, but also more useful for practical candidate prioritization.

Random split models again produced somewhat stronger screening metrics than drug-level split models, which is consistent with the earlier conclusion that random splitting inflates apparent performance. Even so, the XGBoost drug-level split model retained strong early enrichment, making it the most meaningful model for screening-oriented interpretation in this study.

## 7. Feature importance

Feature importance analysis was performed for the XGBoost model trained under the drug-level split, since this was the strongest model under the most realistic evaluation setting.

The importance distribution showed that the model relied overwhelmingly on drug-side features. Approximately 98.15% of total importance was assigned to drug features, while only about 1.85% was assigned to protein features. This indicates that the predictive signal captured by the model comes almost entirely from the ligand representation.

The top 20 individual features were all located in the drug feature block. Most of these corresponded to Morgan fingerprint dimensions, showing that substructure-based chemical information was the main driver of the predictions. One interpretable molecular descriptor, hydrogen bond donors, also appeared among the top-ranked features, suggesting that the model used not only fingerprint patterns but also a small amount of explicit physicochemical information.

This result is informative for two reasons. First, it confirms that the handcrafted drug representation is strong enough to support useful DTI prediction in a classical ML setting. Second, it highlights an important limitation of the current feature design: the protein representation is likely too simple to contribute much additional information. Because the target was encoded only through amino acid composition and sequence length, the model had limited access to higher-order protein characteristics such as motifs, domains, or structural context.

Overall, the feature importance analysis suggests that the current pipeline behaves more like a ligand-driven interaction model than a balanced drug–target model. This does not invalidate the results, but it does define an important direction for future improvement.

## 8. Limitations and future work

This project has several important limitations. The first is the simplicity of the protein representation. Protein targets were encoded only through amino acid composition and sequence length, which provides a coarse description of the sequence but does not capture local motifs, residue order, structural organization, or binding-site information. This limitation is also reflected in the feature importance analysis, where protein features contributed very little compared with drug-side features.

A second limitation is that the models were trained on handcrafted descriptors rather than richer learned representations. This was an intentional design choice, since the purpose of the project was to establish strong classical baselines, but it also means that some potentially useful structural and sequence information was not exploited. In future work, the pipeline could be extended with more expressive protein encodings, improved molecular descriptors, or learned embeddings, while still keeping the classical baseline results as a reference point.

Another limitation is that the evaluation focused mainly on unseen-drug generalization, but not on other challenging settings such as unseen-target or fully cold-start drug–target splits. These settings would provide an even stricter test of generalization and would be useful for understanding how robust the models are across different deployment scenarios.

Despite these limitations, the project successfully demonstrates the main methodological point: evaluation strategy strongly affects the apparent quality of DTI models. Random splits produce inflated estimates, while drug-level splits provide a more realistic view of generalization. Future work should therefore focus not only on improving model architecture, but also on adopting evaluation settings that better match real screening conditions.