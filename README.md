<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&duration=3000&pause=900&color=1F4E79&center=true&vCenter=true&width=700&lines=Risk+Alert+Classifier;Early+Warning+System+for+Banking;Imbalanced+Data+%2B+Ensemble+Learning;Powered+by+Random+Forest+%2B+SMOTE" alt="Typing SVG" />

<br>

![Project Banner](https://capsule-render.vercel.app/api?type=waving&color=0:2E74B5,100:1F4E79&height=200&section=header&text=Risk%20Alert%20Classifier&fontSize=42&fontColor=ffffff)

<br>

[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge&logo=checkmarx&logoColor=white)](#)
[![Model](https://img.shields.io/badge/Final%20Model-Tuned%20Random%20Forest-2E74B5?style=for-the-badge&logo=scikitlearn&logoColor=white)](#)
[![Domain](https://img.shields.io/badge/Domain-Digital%20Banking%20%2F%20Fraud%20Detection-1F4E79?style=for-the-badge&logo=shield&logoColor=white)](#)
[![Type](https://img.shields.io/badge/Type-Binary%20Classification-6B6B6B?style=for-the-badge)](#)

<br>

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter%20Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)

</div>


## 🎬 Video Walkthrough

<div align="center">

[![Watch the Project Walkthrough](https://img.shields.io/badge/▶%20Watch%20Full%20Walkthrough-Google%20Drive-4285F4?style=for-the-badge&logo=google-drive&logoColor=white)](https://drive.google.com/file/d/1yR5E9_BT7HF8LrNZF_bs5nRINXy1QSxs/view?usp=sharing)

> 🎥 **A complete end-to-end video explanation** of the project — covering data preparation, model building, imbalance handling, and the final deployment recommendation.
>
> 📌 *Click the button above or [open the video directly →](https://drive.google.com/file/d/1yR5E9_BT7HF8LrNZF_bs5nRINXy1QSxs/view?usp=sharing)*

</div>

<br>

## 🎯 About the Project

> An end-to-end **classification pipeline** built for a digital banking platform to identify high-risk customers likely to default on payments or engage in fraudulent behavior — combining class imbalance handling, ensemble learning, and hyperparameter tuning.

This project takes on the role of a **Data Scientist at a digital banking platform**, working with customer demographic, behavioral, and transaction data. The challenge goes beyond building an accurate model — it requires correctly handling severe class imbalance, comparing multiple classifiers, and selecting the best model based on a business priority: **minimizing missed high-risk customers (false negatives).**

<div align="center">

```
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
   88% Low Risk   ████████████████████████████░░
   12% High Risk  ████░░░░░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

</div>

<br>

## ✨ Highlights

<table>
<tr>
<td width="50%" valign="top">

### 📐 Three Classifier Approaches
- **Linear baseline** → Logistic Regression
- **Tree-based** → Decision Tree
- **Ensemble** → Random Forest (Tuned)

</td>
<td width="50%" valign="top">

### ⚖️ Imbalance Handling Techniques
- Under-Sampling and Over-Sampling
- SMOTE — synthetic minority oversampling
- ADASYN — adaptive boundary-focused generation

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🔬 Robust Hyperparameter Tuning
- Randomized Search CV for broad parameter exploration
- Grid Search CV for fine-tuning around best region
- Recall-optimized scoring throughout

</td>
<td width="50%" valign="top">

### 🏢 Business-First Thinking
- Type-I vs Type-II error framed in banking context
- Recall prioritized over Accuracy
- Deployment recommendation with justification

</td>
</tr>
</table>

<br>

## 🗺️ Project Journey

<div align="center">

| Stage | Focus | Outcome |
|:---:|:---|:---|
| 🧩 **Part A** | Conceptual Foundations | Logistic Regression, metrics, imbalance explained |
| 📦 **Part B** | Data Preparation | Feature engineering, encoding, KNN imputation, scaling |
| 📍 **Part C** | Baseline Model | Logistic Regression with full metric evaluation |
| 🔄 **Part D** | Imbalance Handling | Under-sampling, SMOTE, ADASYN applied and compared |
| 🌲 **Part E** | Tree-Based Models | Decision Tree vs Random Forest, overfitting analysis |
| ⚙️ **Part F** | Hyperparameter Tuning | Randomized Search + Grid Search for both tree models |
| 📊 **Part G** | ROC Analysis | AUC-ROC curves for all models head-to-head |
| 📋 **Part H** | Reporting | Final model selection, business interpretation |

</div>

<br>

## 🔍 The Dataset

<div align="center">

<table>
<tr><th>Category</th><th>Features</th></tr>
<tr><td>👤 Demographics</td><td>Age, Gender, Region, Employment Type</td></tr>
<tr><td>💰 Financial Profile</td><td>Annual Income, Credit Score, Credit Utilization Ratio, Debt Balance</td></tr>
<tr><td>⚠️ Payment Behavior</td><td>Missed Payments (12m), Avg Late Payment Days, Cash Advances (6m)</td></tr>
<tr><td>📊 Transaction Signals</td><td>Monthly Transaction Count, Monthly Spend, Last Transaction Date → Days Ago</td></tr>
<tr><td>🚨 Risk Signals</td><td>Complaints (6m), Failed Login Attempts (3m), Account Tenure</td></tr>
<tr><td>🎯 Target</td><td><code>risk_status</code> — Low Risk (0) vs High Risk (1)</td></tr>
</table>

</div>

**Dataset size:** 4,600 customer records | **Class split:** ~88% Low Risk, ~12% High Risk

<br>

## 🏆 Final Verdict

<div align="center">

### 🌲 Tuned Random Forest — Recommended for Deployment

</div>

> All four models — Logistic Regression, Decision Tree, untuned Random Forest, and Tuned Random Forest — were evaluated. Logistic Regression achieved the highest scores on this particular test split (Recall and AUC near 1.0), but **Tuned Random Forest was selected as the final deployment model.**

<div align="center">

| Why Tuned Random Forest Leads the Recommendation |
|:---|
| 🌲 Ensemble of 100+ trees — less sensitive to any single train-test split |
| ⚙️ Hyperparameter tuning via Randomized + Grid Search optimized for Recall |
| 📉 Reduces overfitting seen in the untuned Decision Tree (100% train accuracy) |
| 🏦 More robust and reliable for real-world banking deployment |

</div>

<br>

## ⚖️ Why Imbalance Handling Matters

<div align="center">

<table>
<tr>
<td align="center" width="33%">

### 📉 Without Handling
**Model predicts Low Risk always**
<br>
88% accuracy — but 0% High Risk caught

</td>
<td align="center" width="33%">

### 🔄 SMOTE Applied
**Synthetic minority samples generated**
<br>
Balances classes without data loss

</td>
<td align="center" width="33%">

### 🎯 ADASYN Applied
**Focuses on boundary-adjacent samples**
<br>
More aggressive minority generation

</td>
</tr>
</table>

</div>

<div align="center">

**For a banking risk system, missing a high-risk customer is far costlier than a false alarm — so Recall is the primary metric.**

</div>

<br>

## 🎯 Precision vs Recall — The Business Trade-off

<div align="center">

<table>
<tr>
<td align="center" width="50%">

### 🎯 High Precision
**Fewer genuine customers wrongly flagged**
<br>
Protects user experience and reduces unnecessary reviews

</td>
<td align="center" width="50%">

### 🛟 High Recall
**Fewer high-risk customers missed**
<br>
Prevents payment defaults, fraud losses, regulatory risk

</td>
</tr>
</table>

</div>

<div align="center">

**Recall is prioritized: a missed high-risk customer costs more than a false alarm.**

</div>

<br>

## 🧪 Theory Covered

<details>
<summary><b>📖 Click to expand the conceptual foundations explored in this project</b></summary>
<br>

- Logistic Regression and the sigmoid function for binary probability output
- Why accuracy is insufficient for imbalanced datasets
- Type-I Error (False Positive) and Type-II Error (False Negative) in banking risk
- Precision, Recall, F1-Score, TPR, and FPR — definitions and trade-offs
- AUC-ROC as a threshold-independent evaluation metric
- Class imbalance — causes, consequences, and solutions
- Decision Tree overfitting and how depth constraints help
- Random Forest bagging — why averaging across trees reduces variance
- SMOTE vs ADASYN for synthetic minority class generation
- Randomized Search CV vs Grid Search CV for hyperparameter optimization

</details>

<br>

## 🛠️ Tech Stack

<div align="center">

![scikit-learn](https://img.shields.io/badge/scikit--learn-LR%20%7C%20DT%20%7C%20RF%20%7C%20GridSearch-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE%20%7C%20ADASYN-2E74B5?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Operations-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-ROC%20Curve%20Visualization-11557C?style=flat-square)

</div>

<br>

## 📚 What This Project Demonstrates

<div align="center">

```
✔ Handling severely imbalanced datasets using SMOTE and ADASYN
✔ Comparing a linear baseline against tree-based and ensemble models
✔ Diagnosing overfitting through train vs test accuracy gap analysis
✔ Applying Randomized Search + Grid Search for recall-optimized tuning
✔ Evaluating models with AUC-ROC across decision thresholds
✔ Selecting a deployment model based on business risk priorities
```

</div>

<br>

## 👨‍💻 Author

<div align="center">

<img src="https://avatars.githubusercontent.com/u/00000000?v=4" width="100" height="100" style="border-radius: 50%;" alt="Author Avatar"/>

### **Ayush Isamaliya**
*Data Science & Aspiring ML Engineer*

</div>

---

### 🌐 Connect With Me

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-isamaliya16-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/isamaliya16)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush_isamaliya-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ayush-isamaliya-686533312/)

</div>

<br>

<div align="center">

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:1F4E79,100:2E74B5&height=120&section=footer)

**Built as part of the Python Data Science track at Red & White Skill Education**

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=16&duration=4000&pause=1500&color=6B6B6B&center=true&vCenter=true&width=500&lines=Thank+you+for+reviewing+this+project!;Star+%E2%AD%90+if+you+found+it+insightful." alt="Footer Typing SVG" />

*Made with ❤️ | PR3 — Risk Alert Classifier | Imbalanced Classification | Digital Banking Security*

</div>
