# Emre - KNN vs Random Forest + EDA

This folder has my whole work for this task in one place:
- notebook: `emre-knn-random-forest.ipynb`
- summaries: `eda_summary.json`, `metrics_summary.json`
- plots: `figures/`

I used:
- `data/raw/Training.csv`
- `data/raw/Testing.csv`

Data format quick note:
- 132 symptom columns (binary 0/1)
- target = `prognosis` (text label)
- 41 disease classes

## EDA summary

Main checks:
- missing values
- class balance
- symptoms per record
- symptom prevalence
- symptom correlation
- disease-symptom pattern heatmap

Key findings:
- Missing values: 0 in both train and test
- Class balance: perfectly balanced (120 rows per disease in training)
- Symptoms per record: min 3, max 17, mean 7.45, median 6
- Feature prevalence mean: 0.056 (overall sparse feature space)

EDA plots:
- ![EDA Missing Values](figures/eda_missing_values.png)
- ![EDA Class Balance](figures/eda_class_balance.png)
- ![EDA Symptoms Per Record Distribution](figures/eda_symptoms_per_record_distribution.png)
- ![EDA Feature Prevalence Histogram](figures/eda_feature_prevalence_histogram.png)
- ![EDA Top 20 Symptom Correlation](figures/eda_top20_symptom_correlation.png)
- ![EDA Disease Symptom Heatmap](figures/eda_disease_symptom_heatmap.png)
- ![Class Distribution](figures/class_distribution.png)
- ![Top Symptoms](figures/top_symptoms.png)

## Model results

Both models are trained separately (standalone), then compared on the same test set.

| Model | Setting | Test Accuracy | Macro F1 |
|---|---|---:|---:|
| KNN | k=1 (best from 5-fold CV) | 1.0000 | 1.0000 |
| Random Forest | n_estimators=500 | 0.9762 | 0.9837 |

Why scores are high:
- dataset is generated and clean
- binary symptom patterns are very structured
- train/test structure is very similar

Model plots:
- ![KNN CV Curve](figures/knn_cv_curve.png)
- ![Model Accuracy Comparison](figures/model_accuracy_comparison.png)
- ![KNN Confusion Matrix](figures/knn_confusion_matrix.png)
- ![Random Forest Confusion Matrix](figures/rf_confusion_matrix.png)
- ![Random Forest Top Features](figures/rf_top_features.png)

## If I improve this next

1. Use real/noisy patient data to test generalization
2. Tune Random Forest more (`max_depth`, `min_samples_leaf`, etc.)
3. Add top-k diagnosis metrics (top-3)
4. Check probability calibration/confidence quality
