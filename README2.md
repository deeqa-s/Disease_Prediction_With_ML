git # Disease Prediction Using Machine Learning 

## Members
### Cohort: 8, Team: ML06
- [Aakash Bajaj](https://github.com/aakash11bajaj)
- [Amena Muzaffar Shumi]()
- [Deeqa Mahamed]()
- [Ecce Djogbenou Epse Houenou](https://github.com/eccehouenou)
- [Emre Ozkan]()
- [Haimeng (James) Wang]()

# Contents 
- [Project Overview](#Project-Overview)  
- [Business Objective](#business-objective) 
    * [Business Question](#business-question)
    * [Stakeholders](#stakeholders)
- [Dataset](#Dataset) 
    * [Details](#Details) 
    * [Variables/Features](#Variables-/-Features)
    * [Target Variable](#target-variable)
- [Potential Risks and Uncertainty](#potential-risks-and-uncertainty) 
- [Methodology](#methodology) 
    * [Modeling Approach](#modeling-approach) 
    * [Planned Workflow](#planned-workflow)
    * [Technical Stack](#technical-stack)
- [Repository Structure](#rep) 
- [Task Log](#task-log)
    * [Task Completed (Week 1)](#task-completed-week-1)
    * [Task (Week 2)](#task-assignment-for-first-half-of-week-2)
- [References](#references)

# Project Overview
This project applies **unsupervised and supervised machine learning methods to predict diseases based on a patient's reported symptoms.** \
The goal is to build a multi-class classifier that can identify which of 41 possible diseases a patient is most likely to have, based on the presence or absence of 132 distinct symptoms.

A key emphasis of this project is **model interpretability** i.e. understanding *which symptoms drive predictions* and *how confident the model is* in distinguishing between similar diseases with high **predictive accuracy**. 

This makes the project relevant to clinical decision support contexts, where explainability is just as important as performance.

---
# Business Objective
Delayed or incorrect diagnosis remains a critical challenge in healthcare, particularly in resource-limited settings where access to specialists and diagnostic tests is constrained. 

This project aims to **develop a machine learning-based symptom classifier** that can assist clinicians and healthcare workers in identifying the most probable disease from a patient's reported symptoms. 

By surfacing likely diagnoses early in the clinical encounter, the model has the potential to support
- faster triage 
- reduces reliance on broad-spectrum testing
- serves as a structured decision aid for less experienced medical staff 

The ultimate objective is to demonstrate how interpretable machine learning can add measurable value to frontline diagnostic workflows.

---
## Business Question

>  **Can we accurately predict a patient's disease from their reported symptoms?** 

>  **Which symptoms are most diagnostically informative to distinguish between diseases?**

### Sub-questions

1. Which machine learning models yield the highest classification accuracy on symptom-based prediction?
2. Which symptoms carry the most predictive power both globally and per disease?
3. Are there groups of diseases with similar symptom profiles that models frequently confuse?
4. Can we build an interpretable model that a clinician could plausibly trust and act on?

---
### Stakeholders

Technical stakeholders:
* Data scientists and ML engineers
* Public health departments

Institutional stakeholders:
* Hospitals: may want to quickly identify and stratify disease risk based on symptoms
* Insurance companies: might want to identify disease type to manage claims or determine coverage
* Patients: quickly get an accurate diagnosis by searching for a few key symptoms to decide whther to seek medical treatment

---

# Dataset 
## Details 

The dataset contains 4920 observations, each representing a patient record.
There are 134 columns, including 132 symptom features, one target variable (prognosis), and one extra column that was later removed (originally an index column from the CSV).
The symptom features are binary (0 = absence, 1 = presence) indicating whether each symptom is observed for a patient.
The target variable prognosis is categorical with 41 unique disease classes.

![alt text](results/Ecce_results/column_info_table.png)

| Property | Details |
|---|---|
| **Source** | [Kaggle – Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning/data) |
| **Author** | kaushil268 |


## Identified Issues and Limitations

#### 1. Synthetic / Rule-Based Data
The dataset does *not* appear to originate from real clinical records. Symptom-to-disease mappings seem algorithmically generated, likely lacking noise, ambiguity, and co-morbidities found in real patients. 

#### 2. Artificially Balanced Classes
Each disease class likely has a uniform or near-uniform number of records. This does not refelct clinical reality of disease prevalence.

#### 3. Binary Encoding Loses Clinical Nuance
All symptoms are encoded as binary 0/1 flags. 
Severity, duration, and interaction patterns are not being considered.

#### 4. No Patient Demographics
There are no features for age, sex, geographic region, or medical history.

#### 5. High Dimensionality with Potentially Sparse Features
With 132 binary features, many symptoms may be near-zero variance or irrelevant for most disease classes. Feature selection is necessary to avoid overfitting and to improve model interpretability.

#### 6. No Temporal Information
Each row is an isolated symptom profile. Real diagnosis often depends on how symptoms evolve over time, which this dataset does not capture.

> **Due to the above, we belive that some models may achieve near-perfect accuracy in this environment while failing to generalize to real-world clinical settings.**


---
# Methodology 
## Modeling Approach

Since this dataset is likely near-perfectly separable (symptom combinations map cleanly to diseases in a rule-based way), most models will achieve high raw accuracy. 

The more meaningful question becomes: 

**Which model is most interpretable and clinically trustworthy?** 

>Our evaluation will therefore emphasize both performance *and* explainability.

###  Model Choices and their Advantages 

| Model | Key Advantage|
|---|---|
Logistic Regression| Simple, Fast, Easy to Implement 
Decision Tree | Visualizable and Easy to Communicate
Bernoulli Naive Bayes | Most Appropriate for Binary Features
Random Forest | Robust, Handels Binary Features Well
XGBoost | High Interpretability, Strong performance 
K-Nearest Neighbors (KNN)| Non Paramteric Comparitor, Highly Interpretable  

## Planned Workflow
| Phase | Tasks |
|---|---|
| **EDA** | Class distribution, Symptom frequency, Symptom Heatmap, Symptom-disease Association Matrix |
| **Preprocessing** |  Feature selection, Label encoding of `prognosis` |
| **Modeling** | Train all models listed above; use 5-fold CV on training data |
| **Evaluation** | Accuracy, Confusion matrix, Per-class metrics; Compare all models in a summary table |
| **Interpretability** | Feature importances, SHAP values, Result visualizations |
| **Reporting** | Discuss accuracy results, Highlight most predictive symptoms, Examine hardest-to-separate disease pairs, Discuss clinical limitations |


## Technical stack
- **scikit-learn**: Fit and evaluate supervised and unsupervised ML models
- **pandas**: Load and explore the dataset
- **numpy**: Basic data manipulations
- **matplotlib**: Create visualizations.

## Repository Structure

```
Disease_Prediction_With_ML/
│
├── data/
│   ├── processed
│   └── raw/
│       ├── Training.csv
│       └── Testing.csv
│
├── experiments/
│
├── models/
│
├── notebooks/
│   ├── 01_.ipynb
│   ├── 02_.ipynb
│   ├── 03_.ipynb
│   └── 04_.ipynb
│
├── src/
│
├── results/
│
└── README.md
```
Breif Description of the folders: 
* **Data:** Contains the raw data. 
* **Experiments:** A folder containing ipython notebook for data exploration and experiments.
* **Models:** A folder containing the final trained model
* **Images:** Contain all images used in the README.md file
* **README:** This file!

# Task Log
## Task Completed (Week 1)
Selecting the problem statement and dataset: Whole team
Creating git repository: Deeqa
Data exploration: Amena, Deeqa, Ecce
Readme file: Aakash
git support: Emre
Data review and discussion: Whole team
Discussion on readme file and next steps: Aakash, Amena, Ecce, Emre

## Task assignment (for first half of week 2)
Each team member will explore and compare multiple models. They will prepare a report with model code, interpretation, visualizations, evaluations and comparison. 
We have listed the model choices below for each team member.
- Aakash Bajaj: XGBoost, K-Nearest Neighbors (KNN)
- Amena Muzaffar Shumi: Logistic Regression, Random Forest
- Deeqa Mahamed: Decision Tree, XGBoost
- Ecce Djogbenou Epse Houenou: Bernoulli Naive Bayes, Random Forest
- Emre Ozkan: Random Forest, K-Nearest Neighbors (KNN)
- Haimeng (James) Wang: Bernoulli Naive Bayes, Decision Tree


## References

## Exploratory Data Analysis
Data set overview
- The Training dataset contains 1 target variable, `prognosis`  and 132 feature variables
There are a total of 4920 observations.
The 132 features are binary (0 = absence, 1 = presence).
There are no missing values in any variable
There are 94 % of duplicates.
[text](results/emre_results/eda_summary.json)

- The testing data set from the kaggle, contains 42 observations (patients) and 133 columns. Each row represents an individual patient record, while the columns represent symptom indicators and the target disease label.

[text](results/emre_results/metrics_summary.json)
## Target Variable

The `prognosis` column within the dataset contains the disease labels and is our Target variable. The 41 classes span infectious diseases are all perfectly balanced and contained each 120 observations. 
![alt text](results/emre_results/eda_class_balance.png)

Most individuals report between 3 and 6 symptoms, with fewer cases exhibiting a high symptom burden (>14 symptoms). This suggests moderate variability in symptom presentation across patients.


![alt text](results/Ecce_results/avg_symptoms_per_disease.png)

Symptoms such as fatigue,vomiting, high fever appear frequently across multiple diseases.

![alt text](results/james_results/top_symptoms.png)

 Certain diseases show strong associations with specific symptoms, which may help machine learning models distinguish between conditions.
 ![alt text](results/emre_results/eda_top20_symptom_correlation.png)


 # Model development and Evaluation
 ## Model 1: Logistic Regression
The experiment was conducted by splitting the dataset into training and test dataset, stratified. The Logistic regression model was used. With an accuracy of The following results were obtained:


| Metric                   | Score |
| ------------------------ | ----- |
| Accuracy                 | 1.00  |
| Precision (Macro Avg)    | 1.00  |
| Recall (Macro Avg)       | 1.00  |
| F1 Score (Macro Avg)     | 1.00  |
| Precision (Weighted Avg) | 1.00  |
| Recall (Weighted Avg)    | 1.00  |
| F1 Score (Weighted Avg)  | 1.00  |
| Test Samples             | 42    |
| Number of Classes        | 22    |

 [text](results/amena_results/logistic_regression_metrics.json)

Logistic regression Visualizations

![alt text](results/amena_results/logistic_regression_confusion_matrix.png)

Logistic Regression: Strengths and Limitations
- Strengths

The Logistic regression model is simple and interpretable  and help to understand how each symptom contributes to predictions. It is Fast training and Works well with binary features. It provides probabilities for each disease prediction. and have a low risk of overfitting.

- Limitations

The Logistic regression model assumes linear relationships and may not capture complex interactions between symptoms. It has a limited handling of complex patterns and struggles with non-linear symptom-disease relationships.It is sensitive to imbalanced data – biased toward diseases with more samples. It performance depends on good preprocessing and feature selection. With small dataset, there is a  very high accuracy which may not generalize to larger or noisier datasets.


## Model 2: Random Forest
Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their predictions.

## Random Forest Performance

| Metric |              Score |
|--------|               -------|
| Accuracy |              0.976 |
| Precision (Macro Avg) | 0.99 |
| Recall (Macro Avg)     | 0.99 |
| F1 Score (Macro Avg)   | 0.98 |
| Precision (Weighted Avg) | 0.99 |
| Recall (Weighted Avg)    | 0.98 |
| F1 Score (Weighted Avg)  | 0.98 |
| Best Hyperparameters | {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100} |
| Best CV Score | 1.0 |

[text](results/amena_results/random_forest_metrics.json)

Random Forest Visualizations

![alt text](results/emre_results/rf_confusion_matrix.png)

With an accuraccy of 0.97619, the following results were obtained:
Muscle pain, itching, etc are the top features noticed.

![alt text](results/amena_results/rf_top_30_features.png)

- Strengths

Random forest captures complex, non-linear relationships and can handle symptom interactions automatically. It handles high-dimensional data and  robust even with many features. It is resistant to overfitting. It helps identify most predictive symptoms.

- Limitations

Random Forest is less interpretable. It is computationally heavier, slower training and more memory usage than simple models. It is sensitive to imbalanced data. It requires setting random seeds for consistent results.
Overfitting is possible with small data.

## Model 3: Bernoulli Naive Bayes
Bernoulli Naive Bayes has been run and here are the results:
## Bernoulli Naive Bayes Performance

| Metric |                 Score |
|--------|                -------|
| Accuracy |                1.00 |
| Precision (Macro Avg)    | 1.00 |
| Recall (Macro Avg)       | 1.00 |
| F1 Score (Macro Avg)     | 1.00 |
| Precision (Weighted Avg) | 1.00 |
| Recall (Weighted Avg)    | 1.00 |
| F1 Score (Weighted Avg)  | 1.00 |

[text](results/Ecce_results/bernoulli_nb_results.json)

Bernoulli Visualizations

![alt text](results/Ecce_results/bernoulli_confusion_matrix.png)

- Strengths

Bernoulli Naives model works well with binary features. It is computationally efficient, even with many features.

It can manage datasets with many symptoms and provides class probabilities, useful for confidence assessment.

It is simple and interpretable

- Limitations

Bernoulli Naives model assumes feature independence. It cannot model interactions between symptoms and is sensitive to zero probabilities. It may underperform with small datasets and probabilities can be skewed if there is few samples per class.

## Some comparison between models

![alt text](results/james_results/model_comparison.png)

