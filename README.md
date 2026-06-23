<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=Robust%20Regression%20Engine&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=Advanced%20Supervised%20Learning%20%7C%20Regularization%20%7C%20Ensemble%20Methods&descAlignY=58&descSize=16" width="100%"/>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Domain-Real%20Estate%20AI-6366f1?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-0ea5e9?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Ridge%20Regression-✓-64748b?style=flat-square"/>
  <img src="https://img.shields.io/badge/Lasso%20Regression-✓-64748b?style=flat-square"/>
  <img src="https://img.shields.io/badge/Random%20Forest-✓-64748b?style=flat-square"/>
  <img src="https://img.shields.io/badge/SVR-✓-64748b?style=flat-square"/>
  <img src="https://img.shields.io/badge/Cross--Validation-✓-64748b?style=flat-square"/>
</p>

</div>

---
---

## 🎬 Video Walkthrough

<div align="center">

[![Watch the Project Walkthrough](https://img.shields.io/badge/▶%20Watch%20Full%20Walkthrough-Google%20Drive-4285F4?style=for-the-badge&logo=google-drive&logoColor=white)](https://drive.google.com/file/d/1IuLyoWia_rZ_TifhcBIesV8PVTcjXg01/view?usp=sharing)

> 🎥 A complete end-to-end video explanation of the project — covering regularization techniques (Ridge & Lasso), cross-validation strategies, tree-based & SVR models, and real estate price prediction insights.
>
> 📌 *Click the button above or [open the video directly →](https://drive.google.com/file/d/1IuLyoWia_rZ_TifhcBIesV8PVTcjXg01/view?usp=sharing)*

</div>

---

## 📌 Table of Contents

- [🎯 Objective](#-objective)
- [🏢 Problem Statement](#-problem-statement)
- [⚙️ Tech Stack](#️-tech-stack)
- [📐 Part A — Conceptual Foundation](#-part-a--conceptual-foundation)
- [📊 Part B — Dataset Preparation](#-part-b--dataset-understanding--preparation)
- [📉 Part C — Regularized Linear Models](#-part-c--regularized-linear-models)
- [🔁 Part D — Cross-Validation Strategies](#-part-d--cross-validation-strategies)
- [🌳 Part E — Tree-Based Regression](#-part-e--tree-based-regression-models)
- [⚡ Part F — Support Vector Regression](#-part-f--support-vector-regression)
- [📈 Part G — Model Comparison & Evaluation](#-part-g--model-comparison--evaluation)
- [📝 Part H — Final Analysis & Reporting](#-part-h--final-analysis--reporting)
- [📦 Deliverables](#-deliverables)
- [📄 License](#-license)

---

## 🎯 Objective

> **Evaluate students' mastery of advanced supervised learning regression techniques** with a strong focus on regularization, model generalization, cross-validation strategy design, and tree-based algorithms.

By completing this project, students will be able to:

- ✅ Build a **production-ready regression pipeline** from scratch
- ✅ Apply **L1 and L2 regularization** to control overfitting
- ✅ Design and compare **four cross-validation strategies**
- ✅ Train and evaluate **linear, tree-based, and kernel-based** regressors
- ✅ Derive **actionable business insights** from model outputs

---

## 🏢 Problem Statement

You are a **Machine Learning Engineer** at a real estate analytics company. The existing house price prediction model suffers from the following issues:

| Issue | Business Impact |
|-------|----------------|
| Overfitting | Poor generalization on new property listings |
| Unstable predictions | Inconsistent pricing across different city datasets |
| No regularization | High variance in coefficient estimates |
| Single model approach | No ensemble fallback or comparison baseline |

### 🎯 Mission

Build a **robust regression pipeline** that applies regularization, validates properly, compares model types, and generalizes reliably on unseen real estate data.


---

## ⚙️ Tech Stack

<div align="center">

| Layer | Tools |
|-------|-------|
| **Language** | Python 3.10+ |
| **ML Framework** | Scikit-Learn 1.4+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Notebook** | Jupyter Lab |
| **Version Control** | Git + GitHub |

</div>

---

## 📐 Part A — Conceptual Foundation

> Core theoretical concepts that form the foundation of this project.

---

### 1️⃣ What is Regularization?

Regularization adds a **penalty term to the loss function** to prevent a model from learning overly complex patterns from training data. It discourages large coefficient values, which are often a sign of overfitting.

**Why it is needed:**
Without regularization, models tend to memorize training data — including noise — and perform poorly on unseen data. Regularization controls this by constraining the model's complexity.

---

### 2️⃣ Ridge (L2) vs Lasso (L1) Regression

| Property | Ridge (L2) | Lasso (L1) |
|----------|-----------|-----------|
| **Penalty Applied To** | Sum of squared coefficients | Sum of absolute coefficients |
| **Feature Selection** | Retains all features — shrinks them | Can zero out irrelevant features |
| **Output Type** | Dense solution | Sparse solution |
| **Handles Multicollinearity** | Yes — distributes weight evenly | May randomly drop correlated features |
| **Best Use Case** | When all features contribute | When only a few features truly matter |
| **Geometric Constraint** | Circular boundary | Diamond-shaped boundary |

---

### 3️⃣ What is Cross-Validation?

Cross-validation is a **model evaluation technique** that splits data into multiple train and test folds to estimate how a model performs on unseen data. It reduces the risk of misleading results from a single random train-test split.

**Why it is important:**
A single split can overestimate or underestimate true performance by chance. Cross-validation averages results across multiple splits for a more reliable, stable estimate.

---

### 4️⃣ Cross-Validation Techniques

#### 🔁 K-Fold Cross-Validation
Data is divided into **K equal folds**. The model trains on K-1 folds and validates on the remaining one, rotating K times. The final score is the average across all K rounds. This is the most commonly used strategy for general-purpose evaluation.

#### ⚖️ Stratified K-Fold Cross-Validation
An extension of K-Fold that **preserves the distribution of target values** in every fold. The continuous target is binned into groups so each fold contains a balanced mix of all price ranges — preventing folds dominated by only high-price or only low-price houses.

#### 🔬 Leave-One-Out Cross-Validation (LOOCV)
A special case where **K equals the total number of samples**. Each sample is used as the sole validation point once. Produces a nearly unbiased estimate but is computationally expensive — practical only for small datasets.

#### 📅 Time Series Split
Strictly **respects time order** — training always uses earlier data, validation always uses later data. Prevents future data from leaking into historical training windows, which is essential for any time-dependent dataset.

---

### 5️⃣ Why Tree-Based Models Don't Need Feature Scaling

Tree-based models make decisions by applying **threshold splits on individual features** — for example, "Is square footage greater than 1,500?" These comparisons use relative values, not absolute magnitudes or distances. Scaling does not change the order or relative ranking of values, so it has no effect on split points or model output.

Linear models and SVR rely on distances or dot products, where differences in feature scale directly distort results — making normalization essential for them.

---

## 📊 Part B — Dataset Understanding & Preparation

### Dataset Overview

| Feature Category | Description |
|-----------------|-------------|
| **Structural Attributes** | Square footage, number of rooms, floors, garage spaces |
| **Location Indicators** | Proximity scores, neighborhood index, area rating |
| **Temporal Details** | Year built, year renovated, property age |
| **Target Variable** | House Price (USD) — continuous numeric value |

---

### Task 6 — Feature & Target Identification

The **target variable** is House Price. All remaining columns serve as input features. A preliminary review identifies the data type of each column and flags any that require special treatment such as encoding or date parsing.

---

### Task 7 — Train-Test Split

| Split | Proportion | Purpose |
|-------|-----------|---------|
| **Training Set** | 80% | Model fitting and hyperparameter tuning |
| **Test Set** | 20% | Final evaluation only — never seen during training |

A fixed random seed ensures the split is reproducible across all experiments.

---

### Task 8 — Preprocessing & Feature Scaling

Feature scaling is applied so linear models and SVR are not distorted by differences in feature magnitude. A **StandardScaler** (zero mean, unit variance) is fitted only on the training set. The same transformation is then applied to the test set — fitting on test data would constitute data leakage.

> **Note:** Tree-based models do not require scaling. It is only applied for Ridge, Lasso, and SVR.

---

## 📉 Part C — Regularized Linear Models

### Task 9 — Ridge Regression (L2)

Ridge adds the **sum of squared coefficients** as a penalty to the loss function. This shrinks all coefficients toward zero but never eliminates any feature entirely. It is particularly effective when many features are correlated — a common scenario in real estate data where size, rooms, and area often move together.

**Key parameter — Alpha (α):** Controls regularization strength. Higher alpha means more shrinkage. Optimal alpha is found using cross-validation.

---

### Task 10 — Lasso Regression (L1)

Lasso adds the **sum of absolute coefficient values** as a penalty. Unlike Ridge, Lasso can reduce some coefficients to exactly zero, removing those features from the model. This makes Lasso a powerful tool for automatic feature selection.

**Key insight:** If Lasso zeroes out a feature's coefficient, that feature has no meaningful predictive value for house prices — directly actionable for the business team.

---

### Task 11 — Alpha Tuning via Cross-Validation

The regularization parameter alpha is tuned by testing a wide logarithmic range of values and selecting the one that produces the best average validation score. This process is automated using built-in cross-validated estimators for both Ridge and Lasso.

---

### Task 12 — Ridge vs Lasso Comparison

| Aspect | Ridge (L2) | Lasso (L1) |
|--------|-----------|-----------|
| **Training Error** | Low — fits training data well | Slightly higher due to sparsity |
| **Validation Error** | Stable across different splits | May vary with feature sparsity |
| **Coefficient Behavior** | All features retain non-zero values | Some features completely zeroed out |
| **Feature Selection** | No automatic selection | Built-in automatic selection |
| **Interpretability** | Moderate | High — clear feature importance |
| **Multicollinearity** | Handles well | May randomly drop correlated features |

---

## 🔁 Part D — Cross-Validation Strategies

### Task 13 — All Four CV Techniques Applied

Each strategy is applied to the same model and dataset for a fair comparison. Every technique produces a set of R² scores — one per fold — which are averaged and analyzed for consistency.

#### K-Fold Cross-Validation
- Splits data into 10 equal folds with shuffling enabled
- Each fold serves as the validation set exactly once
- Average R² across all 10 folds is the reported performance metric
- Standard baseline CV method for most regression tasks

#### Stratified K-Fold Cross-Validation
- House price is first binned into 5 quantile groups
- Ensures each fold contains a proportional mix of all price ranges
- Prevents folds dominated by only high-price or low-price samples
- Produces a more representative and fair evaluation

#### Leave-One-Out Cross-Validation (LOOCV)
- Number of folds equals total number of samples
- Provides the least biased performance estimate possible
- Very high computational cost — best for small datasets only
- Results typically align closely with K-Fold when K ≥ 10

#### Time Series Split
- Training always uses earlier data; validation uses later data
- Prevents future price data from leaking into training windows
- Essential for datasets with temporal features or price trends
- Reflects realistic deployment conditions most accurately

---

### Task 14 — CV Strategy Performance Analysis

| CV Strategy | Bias | Variance | Compute Cost | Best For |
|-------------|------|----------|-------------|---------|
| K-Fold (k=10) | Low | Moderate | Low | General-purpose evaluation |
| Stratified K-Fold | Low | Low | Low | Skewed target distributions |
| LOOCV | Very Low | High | Very High | Small datasets only |
| Time Series Split | Moderate | Moderate | Low | Time-ordered data |

---

## 🌳 Part E — Tree-Based Regression Models

### Task 15 & 16 — Decision Tree Regression

A Decision Tree learns by splitting data at thresholds that minimize prediction error. Without constraints, a fully grown tree memorizes training data and overfits severely. Complexity is controlled through hyperparameters.

| Hyperparameter | Purpose | Effect of Increasing |
|---------------|---------|---------------------|
| **max_depth** | Limits how deep the tree grows | More complex — higher overfit risk |
| **min_samples_split** | Minimum samples needed to split a node | Simpler tree — less overfitting |
| **min_samples_leaf** | Minimum samples required at a leaf | Smoother predictions — reduced variance |

> **Pruning Strategy:** Start shallow and increase depth gradually while monitoring validation error. Stop when validation error stops improving.

---

### Task 17 — Random Forest Regression

A Random Forest is an ensemble of many Decision Trees, each trained on a **random bootstrap sample** and a **random feature subset** at every split. Final predictions are the average across all trees — dramatically reducing variance without increasing bias.

| Hyperparameter | Typical Value | Purpose |
|---------------|-------------|---------|
| **n_estimators** | 100–500 | Number of trees in the forest |
| **max_depth** | 8–15 | Maximum depth per tree |
| **max_features** | sqrt(n_features) | Feature subset per split |
| **min_samples_split** | 10–20 | Minimum samples to allow a split |
| **bootstrap** | True | Random sampling with replacement |

---

### Task 18 — Single Tree vs Ensemble Performance

| Model | Train R² | Test R² | Overfit Risk | Interpretability |
|-------|----------|---------|-------------|----------------|
| Decision Tree (deep) | Very High | Low | High | High — fully visualizable |
| Decision Tree (pruned) | Moderate | Moderate | Medium | High — readable tree |
| Random Forest | High | High | Low | Moderate — feature importance |

> **Key Insight:** Ensemble averaging reduces prediction variance without meaningfully increasing bias — this is the core advantage of Random Forest over a single tree.

---

## ⚡ Part F — Support Vector Regression

### Task 19 — SVR with Multiple Kernels

SVR finds a function that fits as many data points as possible within a margin (epsilon tube) around the prediction. Points outside the tube are penalized. Different kernel functions allow SVR to model linear and non-linear relationships.

| Kernel | How It Works | Strength | Weakness |
|--------|-------------|---------|---------|
| **Linear** | Fits a straight hyperplane | Fast, interpretable | Cannot capture non-linear patterns |
| **RBF (Gaussian)** | Maps data into higher-dimensional space | Excellent for non-linear data | Sensitive to C and γ |
| **Polynomial** | Uses polynomial feature combinations | Flexible boundary | Risk of overfitting; slower |

---

### Task 20 — Hyperparameter Tuning

| Hyperparameter | Role | Typical Search Range |
|---------------|------|---------------------|
| **C (Regularization)** | Penalty for errors outside the epsilon tube. Higher C = less tolerance. | 0.1 → 1000 |
| **γ (Gamma)** | Influence reach of each training sample in RBF. Low γ = broad influence. | 0.001 → auto |
| **ε (Epsilon)** | Width of the tube where errors are not penalized | 0.01 → 1.0 |

---

### Task 21 — SVR vs Other Models

| Aspect | SVR (RBF) | Ridge / Lasso | Random Forest |
|--------|----------|--------------|--------------|
| **Handles Non-linearity** | Yes — kernel trick | No — linear only | Yes — tree splits |
| **Feature Scaling** | Required | Required | Not required |
| **Interpretability** | Low | High — readable coefficients | Moderate |
| **Training Speed** | Slow on large data | Very fast | Moderate |
| **Hyperparameter Sensitivity** | High | Low (alpha only) | Moderate |

---

## 📈 Part G — Model Comparison & Evaluation

### Task 22 — Evaluation Metrics

| Metric | What It Measures | Interpretation |
|--------|----------------|---------------|
| **MSE** | Average of squared prediction errors | Penalizes large errors heavily. Lower is better. |
| **MAE** | Average of absolute prediction errors | Easy to interpret in price units (USD). Lower is better. |
| **RMSE** | Square root of MSE | Same units as house price — more intuitive than MSE. |
| **R² Score** | Proportion of price variance explained | Ranges 0 to 1. Closer to 1 is better. |

---

### Task 23 — Master Model Comparison

| Model | MSE | MAE | RMSE | R² Score | Overfit Risk |
|-------|-----|-----|------|----------|-------------|
| Ridge Regression | — | — | — | — | Low |
| Lasso Regression | — | — | — | — | Low |
| Decision Tree (pruned) | — | — | — | — | Medium |
| Random Forest | — | — | — | — | Low |
| SVR (Linear) | — | — | — | — | Low |
| SVR (RBF) | — | — | — | — | Medium |

> Fill in metric values after running experiments in your notebook.

---

### Task 24 — Overfitting & Underfitting Signals

| Signal | Description | Diagnosis | Solution |
|--------|------------|-----------|---------|
| Train R² >> Test R² | Large gap between sets | **Overfitting** | Regularize, prune, reduce complexity |
| Both R² values are low | Fails on both sets | **Underfitting** | Increase complexity, add features |
| Train R² ≈ Test R² (both high) | Consistent high performance | **Well-generalized** | Ready for deployment |
| High variance across CV folds | Performance varies widely | **Unstable model** | Use ensembles or more training data |

---

## 📝 Part H — Final Analysis & Reporting

### Task 25 — Final Report Coverage

#### 🏆 Best-Performing Model
Determined after comparing all models on the holdout test set. Expected top candidates are **Random Forest** (non-linear datasets) and **Ridge Regression** (high feature correlation). Selection is based on RMSE and R² Score.

---

#### 🔒 Impact of Regularization

| Finding | Detail |
|---------|--------|
| Ridge stabilizes predictions | Prevents coefficient explosion when features are correlated |
| Lasso removes noise features | Automatically zeroes out uninformative predictors |
| Both outperform plain OLS | Regularized models beat unregularized regression on holdout data |
| Alpha selection is critical | Too low causes overfitting; too high causes underfitting |

---

#### 🔁 Role of Cross-Validation in Stability

- K-Fold provides a reliable, efficient general-purpose estimate
- Stratified K-Fold ensures consistent evaluation across all price ranges
- LOOCV confirms near-unbiased estimates for small datasets
- Time Series Split is essential for temporal features to prevent leakage
- CV results directly guide optimal alpha selection for Ridge and Lasso

---

#### 📊 Linear vs Non-Linear Regressors

| Aspect | Linear Models | Non-Linear Models |
|--------|--------------|------------------|
| **Interpretability** | High — coefficients directly readable | Lower — less transparent |
| **Training Speed** | Very fast | Slower |
| **Feature Engineering** | May need manual interaction terms | Captures interactions automatically |
| **Performance** | Good for roughly linear relationships | Better for complex patterns |
| **Stability** | Very stable across datasets | May vary with hyperparameter choices |

---

#### 🏠 Business Interpretation

The **Random Forest** model identifies property size, location index, and renovation year as the top drivers of house price — directly actionable for the company's valuation team.

**Regularized linear models** provide stable pricing baselines ideal for automated listing tools where speed and interpretability are priorities.

**Cross-validation results** confirm that model performance is consistent across all data splits — giving the company confidence to deploy predictions across all city datasets.

---

### Task 26 — Submission Checklist

| Deliverable | Description |
|-------------|-------------|
| ✅ Jupyter Notebooks (Parts A–H) | One notebook per part with clear, reproducible outputs |
| ✅ Evaluation Tables (CSV) | All model metrics consolidated in a single file |
| ✅ Visualization Plots | Coefficient paths, learning curves, residual plots |
| ✅ Final PDF Report | Business-ready summary with insights and recommendations |
| ✅ Clean Source Code | Modular, documented Python modules in `src/` |

---

## 👨‍💻 Author

<div align="center">

<img src="https://avatars.githubusercontent.com/u/00000000?v=4" width="100" height="100" style="border-radius: 50%;" alt="Author Avatar"/>

### **Ayush Isamaliya**
*Data Science  & Aspiring ML Engineer*

</div>

---

---

### 🌐 Connect With Me

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-isamaliya16-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/isamaliya16)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush_isamaliya-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayush-isamaliya-686533312/)

</div>

---
---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=120&section=footer" width="100%"/>

**Robust Regression Engine · Machine Learning Engineering · Real Estate Analytics**

</div>
