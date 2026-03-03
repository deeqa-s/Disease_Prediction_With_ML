# Disease Prediction Using Machine Learning 

## Members
### Cohort: 8, Team: ML06
- [Aakash Bajaj](https://github.com/aakash11bajaj)
- [Amena Muzaffar Shumi]()
- [Deeqa Mahamed]()
- [Ecce Djogbenou Epse Houenou]()
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
* Data scientists and ML engineers:
* Public health departments

Institutional stakeholders:
* Hospitals: may want to quickly identify and stratify disease risk based on symptoms
* Insurance companies: might want to identify disease type to manage claims or determine coverage
* Patients: quickly get an accurate diagnosis by searching for a few key symptoms to decide whther to seek medical treatment

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

The datatset contains 132 binary symptom features. Each recording if the symptom is present or absent. 

Example include: `itching`, `skin_rash`, `nodal_skin_eruptions`, `continuous_sneezing`, `shivering`, `chills`, `joint_pain`, `stomach_pain`, `acidity`, `vomiting`, `fatigue`, `weight_gain`, `anxiety`, `cold_hands_and_feets`, `mood_swings`, `weight_loss`, `restlessness`, `lethargy`, `patches_in_throat`, `irregular_sugar_level`, and many others.

## Target Variable

The `prognosis` column within the dataset contains the disease labels and is our Target variable. Each row has one of 41 potential labels.

The 41 classes span infectious diseases (e.g., Typhoid, Dengue, Malaria), chronic conditions (e.g., Diabetes, Hypertension, Arthritis), skin disorders (e.g., Psoriasis, Fungal infection), and gastrointestinal/respiratory conditions.

---
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

