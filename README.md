
# Disease Prediction Using Machine Learning 

## Members
### Cohort: 8, Team: ML06
- [Aakash Bajaj](https://github.com/aakash11bajaj)
- [Amena Muzaffar Shumi](https://github.com/amena2712-del)
- [Deeqa Mahamed](https://github.com/deeqa-s)
- [Ecce Djogbenou Epse Houenou](https://github.com/eccehouenou)
- [Emre Ozkan]()
- [Haimeng (James) Wang](https://github.com/hmwang-ca)
---
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

---
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

**Technical stakeholders:**
* Data scientists and ML engineers:
* Public health departments

**Institutional stakeholders:**
* Hospitals: may want to quickly identify and stratify disease risk based on symptoms
* Insurance companies: might want to identify disease type to manage claims or determine coverage
* Patients: quickly get an accurate diagnosis by searching for a few key symptoms to decide whether to seek medical treatment

---

# Dataset 
## Details 

| Property | Details |
|---|---|
| **Source** | [Kaggle – Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning/data) |
| **Author** | kaushil268 |
| **Files** | `Training.csv`, `Testing.csv` |
| **Total Columns** | 133 |
| **Feature Columns** | 132 (binary symptom indicators: 0 = absent, 1 = present) |
| **Target Column** | `prognosis` (disease label) |
| **Training Rows** | 4,920 |
| **Testing Rows**  | 42 |
| **Number of Disease Classes** | 41 |

##  Variables / Features 

The dataset contains 132 binary symptom features. Each recording if the symptom is present or absent. 

Example include: `itching`, `skin_rash`, `nodal_skin_eruptions`, `continuous_sneezing`, `shivering`, `chills`, `joint_pain`, `stomach_pain`, `acidity`, `vomiting`, `fatigue`, `weight_gain`, `anxiety`, `cold_hands_and_feets`, `mood_swings`, `weight_loss`, `restlessness`, `lethargy`, `patches_in_throat`, `irregular_sugar_level`, and many others.

## Target Variable

The `prognosis` column within the dataset contains the disease labels and is our Target variable. Each row has one of 41 potential labels.

The 41 classes span infectious diseases (e.g., Typhoid, Dengue, Malaria), chronic conditions (e.g., Diabetes, Hypertension, Arthritis), skin disorders (e.g., Psoriasis, Fungal infection), and gastrointestinal/respiratory conditions.

## Identified Issues and Limitations

#### 1. Synthetic / Rule-Based Data
The dataset does *not* appear to originate from real clinical records. Symptom-to-disease mappings seem algorithmically generated, likely lacking noise, ambiguity, and co-morbidities found in real patients. 

#### 2. Artificially Balanced Classes
Each disease class likely has a uniform or near-uniform number of records. This does not reflect clinical reality of disease prevalence.

#### 3. Binary Encoding Loses Clinical Nuance
All symptoms are encoded as binary 0/1 flags. 
Severity, duration, and interaction patterns are not being considered.

#### 4. No Patient Demographics
There are no features for age, sex, geographic region, or medical history.

#### 5. High Dimensionality with Potentially Sparse Features
With 132 binary features, many symptoms may be near-zero variance or irrelevant for most disease classes. Feature selection is necessary to avoid overfitting and to improve model interpretability.

#### 6. No Temporal Information
Each row is an isolated symptom profile. Real diagnosis often depends on how symptoms evolve over time, which this dataset does not capture.

> **Due to the above, we believe that some models may achieve near-perfect accuracy in this environment while failing to generalize to real-world clinical settings.**

>**_Results should be interpreted accordingly._**

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
Random Forest | Robust, Handles Binary Features Well
XGBoost | High Interpretability, Strong performance 
K-Nearest Neighbors (KNN)| Non Parametric Comparator, Highly Interpretable  

## Planned Workflow
| Phase | Tasks |
|---|---|
| **EDA** | Class distribution, Symptom frequency, Symptom Heatmap, Symptom-disease Association Matrix |
| **Preprocessing** |  Feature selection, Label encoding of `prognosis` |
| **Modeling** | Train all models listed above; use 5-fold CV on training data |
| **Evaluation** | Accuracy, Confusion matrix, Per-class metrics; Compare all models in a summary table |
| **Interpretability** | Feature importance's, SHAP values, Result visualizations |
| **Reporting** | Discuss accuracy results, Highlight most predictive symptoms, Examine hardest-to-separate disease pairs, Discuss clinical limitations |


## Technical stack
- **scikit-learn**: Fit and evaluate supervised and unsupervised ML models
- **pandas**: Load and explore the dataset
- **numpy**: Basic data manipulations
- **matplotlib**: Create visualizations

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
Brief Description of the folders: 
* **Data:** Contains the raw data. 
* **Experiments:** A folder containing ipython notebook for data exploration and experiments.
* **Models:** A folder containing the final trained model
* **Images:** Contain all images used in the README.md file
* **README:** This file!

---
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

---
# Exploratory data analysis
### Dataset Overview
The dataset contains 4920 observations (rows), each representing a patient record.
There are 134 columns, including 132 symptom features, one target variable (prognosis), and one extra column that was later removed (originally an index column from the CSV).
The symptom features are binary (0 = absence, 1 = presence) indicating whether each symptom is observed for a patient.
The target variable prognosis is categorical with 41 unique disease classes.

---
# Modeling
## Model 2: Decision Tree Model
The second experiment involved training a Decision Tree model, where the results were as follows: 
| Metric | Value (Decision Tree) |
|---|---|
| 5-Fold CV Accuracy | 1.0000 ± 0.0000 |
| Test Accuracy | **97.62%** |
| Misclassifications | 1 / 42 |
| Tree Depth | 55 |
| Number of Leaves | 69 |

### Decision Tree Model Visualizations

**Decision Tree - First 3 Levels**
<p align="center">
  <img src="results/james_results/dt_tree.png" width="750" height = "400">
</p>

**Confusion Matrix - Decision Tree (Test Set)**
<p align="center">
  <img src="results/james_results/confusion_matrix_dt.png" width="750" height = "400">
</p>

**Top 20 Feature Importance's — Decision Tree**
<p align="center">
  <img src="results/james_results/dt_feature_importance.png" width="750" height = "400">
</p>

### Decision Tree Strengths and Limitations  

**Strengths:**
- Highly interpretable — decision rules are fully traceable from root to leaf.
- No feature independence assumption; handles correlated symptoms naturally.
- Feature importance clearly identifies `high_fever` and `yellowing_of_eyes` as the most discriminative symptoms.

**Limitations:**
- Very deep tree (depth = 55) indicates the model has memorized the training data — a hallmark of overfitting on this synthetic dataset.
- In real-world data, an unpruned tree this deep would generalize poorly. Pruning (`max_depth`, `min_samples_leaf`) or using an ensemble (Random Forest) would be advisable.
- The 1 misclassification (Fungal infection) suggests boundary cases exist where symptom overlap confuses the tree.

## Model 5: XGBoost
The fifth experiment involved training an XGBoost model, where the results were as follows:

| Metric | Value (XGBoost) |
|---|---|
| 5-Fold CV Accuracy<sup>*</sup>| 100% |
| Test Accuracy | **97.62%** |
| Test Precision | **98.78%** |
| Best Learning Rate| 0.05 |
| Best Max Depth | 3 |
| Best Number of Estimators | 50 |
<sup>*</sup> Fitting 5 folds for each of 27 candidates, totalling 135 fits.

### XGBoost Visualizations

**Confusion Matrix - XGBoost**
<p align="center">
  <img src="results/aakash_results/confusion_matrix_XGBoost.png" width="750" height = "400">
</p>

**Top 20 Feature importance's — XGBoost**
<p align="center">
  <img src="results/aakash_results/xgb_feature_importance.png" width="750" height = "400">
</p>

**SHAP Analysis — XGBoost**
<p align="center">
  <img src="results/aakash_results/xgb_shap_bar.png" width="750" height = "400">
</p>

### XGBoost Strengths and Limitations  

**Strengths:**
- XGBoost demonstrated strong and consistent performance on this dataset indicating a highly stable and reliable model. 
- On the held-out test set it achieved high accuracy (97.62% ), correctly classifying 41 out of 42 disease cases. 
- The grid search revealed a remarkably shallow but optimal configuration (max_depth=3, n_estimators=50, learning_rate=0.05), suggesting it did not require aggressive complexity to learn the underlying symptom-disease relationships. 
- This model is very  explainable as it provides native feature importance scores and supports SHAP-based interpretability. 
- The SHAP analysis identified the specific symptoms (`family_history` and `mild_fever`) with highest diagnostic impact across all 42 disease classes.

**Limitations:**
- XGBoost model did not achieve perfect test set accuracy, misclassifying 1 out of 42 test cases (97.62%). 
- The model's boosting mechanism may have introduced a marginal degree of overfitting relative to simpler models.
- It also had the longest training time at 5.991 seconds, making it a lot slower than other models for no gain in test accuracy. 

## Model 6: Hierarchical and KNN Clustering
The sixth experiment involved training two clustering algorithms -  Hierarchical Clustering (Unsupervised) and KNN Clustering (Supervised)

### Hierarchical Clustering
For investigating natural clusters in the data, the *Jaccard distance* measure was used as it can accommodate binary features.

**Hierarchical Clustering - Dendrogram**
<p align="center">
  <img src="results/aakash_results/deeqa_HC_dendrogram.png" width="750" height = "400">
</p>
There appear to be ~41 'natural' clusters in the data set that even split into a roughly equal number of observations, matching the number of diseases.

- The *Agglomerative Hierarchical Clustering* model was fit and had an accuracy of **100%**.

**Hierarchical Clustering - Prognosis Cluster Heat Map**
<p align="center">
  <img src="results/aakash_results/deeqa_HC_HeatMap.png" width="750" height = "400">
</p>

### KNN Clustering
The results for KNN Clustering were as follows:

| Metric | Value (KNN) |
|---|---|
| 5-Fold CV Accuracy| 100% |
| Test Accuracy | **100%** |
| Test Precision | **100%** |
| Best k| 1 |
| Best k Accuracy | **100%**  |

### KNN Visualizations

**Confusion Matrix - KNN (k = 5)**
<p align="center">
  <img src="results/aakash_results/confusion_matrix_KNN_(k=5).png" width="750" height = "400">
</p>

**KNN — Macro F1 vs K**
<p align="center">
  <img src="results/aakash_results/knn_k_curve.png" width="750" height = "250">
</p>


### Hierarchical Clustering & KNN Strengths and Limitations    

**Strengths:**
- Both methods are grounded in the same fundamental principle: distance between symptom profiles -
  - KNN uses distance to classify individual patients. 
  - Hierarchical clustering uses distance to reveal the broader structure of disease relationships in symptom space.
  - KNN works perfectly here because the diseases are genuinely distinct, and Hierarchical Clustering shows why and where that distinctiveness exists.
  - Both models were also very fast for *this* dataset.
- *Hierarchical Clustering*
  - The dendrogram revealed that most diseases occupy distinct regions in symptom space, with branch merge heights that confirm low inter-class similarity. 
  - The clustered heatmap further showed that disease groups share coherent symptom signatures.
- *KNN* 
  - Achieved perfect scores (100%) across accuracy, Macro F1, Precision and Recall on both CV and test set
  - Completely robust to hyperparameter choice, all values of K from 1 to 25 and all four distance metrics (Euclidean, Manhattan, Hamming, Jaccard) produced identical perfect results
  - No assumptions made about the underlying data distribution, appropriate for a high-dimensional binary symptom space
  - As a non-parametric method *no* assumptions about the shape of the decision boundary are made, useful in a high-dimensional binary feature space where linear separability cannot be assumed

**Limitations:**
-  Obtaining a clean cluster structure from hierarchical clustering and KNN's perfect performance is a direct reflection of the synthetic, rule-based nature of the dataset. 
- *Hierarchical Clustering*
  - Unsupervised method with no predictive output, it cannot be directly compared to other models
  - Subjective choice of Number of clusters by visual inspection of the dendrogram rather than any formal optimization criterion
  - Memory and compute requirements grow quadratically with dataset size, making it impractical at scale without prior dimensionality reduction
- *KNN* 
  - k=1 identified as optimal, meaning the model functions as a lookup table rather than a generalizing classifier 
  -  No native interpretability, it cannot explain which symptoms drove a specific prediction, only which training cases were most similar
  - Prediction time scales linearly with training set size — becomes impractical at the scale of real hospital records without engineering investment

## Model Comparisons
After fitting and evaluating the 6 Models described above, we have the following results 

**Metric Comparisons**
<div align="center">

| Model | CV Accuracy | CV Acc Std | CV Macro F1 | CV F1 Std | Test Accuracy | Test Macro F1 | Test Precision | Test Recall | Train Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 100% | 0% | 100% | 0% | 100% | 100% | 100% | 100% | 0.250 |
| Bernoulli Naive Bayes | 100% | 0% | 100% | 0% | 100% | 100% | 100% | 100% | 0.034 |
| KNN | 100% | 0% | 100% | 0% | 100% | 100% | 100% | 100% | 0.005 |
| Decision Tree | 100% | 0% | 100% | 0% | 97.62% | 98.37% | 98.78% | 98.78% | 0.042 |
| Random Forest | 100% | 0% | 100% | 0% | 97.62% | 98.37% | 98.78% | 98.78% | 0.216 |
| XGBoost | 100% | 0% | 100% | 0% | 97.62% | 98.37% | 98.78% | 98.78% | 3.248 |

</div>

**CV-Accuracy vs Test Accuracy - All Models**
<p align="center">
  <img src="results/comparison_results/accuracy_comparison.png" 
  width="550" height = "200">
</p>

- CV & Test accuracy across different models is not very different.
- Simpler models perform better than complex models for this dataset.

**F1, Precision, Recall on Test set - All Models**
<p align="center">
  <img src="results/comparison_results/f1_precision_recall_comparison.png" 
  width="550" height = "200">
</p>

- All models have nearly identical F1, Precision, Recall on Test set.

**Accuracy vs Training Time - Efficiency tradeoff - All Models**
<p align="center">
  <img src="results/comparison_results/accuracy_vs_time.png" 
  width="550" height = "200">
</p>

- Logistic regression, Bernoulli Naive Bayes and KNN: These models are on the top left, implying highest time efficient and accuracy.  
- Decision Tree and Random Forest: These models are on the bottom left corner, implying lower accuracy but time efficient.
- XGBoost: This model is on the far bottom right, implying low time efficiency and model accuracy.

**Multi-Metric Radar Chart - All Models**
<p align="center">
  <img src="results/comparison_results/radar_chart.png" 
  width="550" height = "350">
</p>

- The Multi metric polygon nearly overlaps for all the models for this dataset. This implies that all models are comparable in performance.

---
#  Conclusions 
- All six models achieved near-perfect or perfect performance, confirming the **dataset is** highly separable and likely **synthetic rule-based in nature**.
- **Simpler, faster models** (Logistic Regression, Bernoulli Naive Bayes, KNN) matched or **outperformed more complex models** (XGBoost, Random Forest, Decision Tree) — complexity adds no value here.
- XGBoost and Decision Tree were the only models to miss a test case, both misclassifying the same "Fungal infection" record — the most ambiguous symptom profile in the dataset.
- Hierarchical Clustering independently validated the separability finding the 41 natural clusters emerging matching the 41 disease classes without using any labels.
- Results **cannot be generalized** to real clinical settings, model performance would degrade significantly with noisy, incomplete, or atypical real-world patient data.

---
# Future Directions
- Test all models on a real clinical dataset 
- Introduce artificial noise to simulate real-world data quality and measure model robustness
- Extend SHAP analysis to per-disease level to identify the most discriminative symptom for each of the 41 conditions individually
- Explore multi-label classification to handle co-morbidities, where a patient may present with more than one condition simultaneously
- Build a simple interactive symptom checker as a proof-of-concept clinical tool on top of the best interpretable model

---
# Team Videos
- [Aakash Bajaj]()
- [Amena Muzaffar Shumi]()
- [Deeqa Mahamed]()
- [Ecce Djogbenou Epse Houenou]()
- [Emre Ozkan]()
- [Haimeng (James) Wang]()

---
# References