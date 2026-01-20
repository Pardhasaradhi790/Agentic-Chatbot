

---

## 1. Business Objective

Accurately estimate the total processing time per request.

**Primary KPI:** R² (explained variance)  
**Secondary KPIs:** RMSE & MAE

---

## 2. Data Sources

- **Claims Table:** `Claims_Rework_ProcessTable last 10 days.csv`  
  *Key columns:* ID,Run_Date,ProcessRunKey,PlanCode,ClaimID,ClaimStatus,specialty,discrepancy_type,StartDate,EndDate,Process_Status,Fallout_Reason,Memo_header,Memo_Description,Action,Remit_Code,DENYcode,CleanDate,Claim_Line,OverrideEdits,CotivityEdits,RequestID,PriorityID,ProjectID

- **Reporting Table:** `ReportingTable_claims_rework last 10 days.csv`  
  *Key columns:* ID (int), Process_Run_Key (varchar), Process_Name (varchar), Task_Name (varchar), Plan_Name (varchar), Run_Date (datetime), Environment (varchar), Start_Time (datetime), End_Time (datetime), BreadCrumbs (varchar), Machine_Name (varchar), Status (varchar), Error_Line (varchar), Error_Desc (varchar), BOT_User_ID (varchar), BOT_Type (varchar), Claim_ID (varchar), Claim_Status (varchar), Adjusted_Claim_ID (varchar), Adjusted_Claim_Status (varchar), FormType (varchar), Memo_Header (varchar), Memo_Message (varchar), CleanDate (datetime), S_Claim_ID (varchar), S_Claim_Status (varchar), S_Reversed_Claim_ID (varchar), S_Reversed_Claim_Status (varchar), EnrollmentType (varchar), PayAmount (varchar), Processed_Claim_ID (varchar), Processed_Claim_Status (varchar)

**Join Key:** `ClaimID` ↔ `Claim_ID` (inner join)

---

## 3. Feature Engineering and final columns considered for training
- **Input Features:**
  - Total_Claims (numeric)
  - PriorityID (label encoded)
  - Each Status type count columns: ADJUCATED, COTIVDISC, DENIED, DENY, OPEN, P737095, P739803, PAID, PAY, PEND, UNKNOWN, VOID
- **Target Variable:**
  - Total_Time_Seconds

``
## 4. Model Performance

### 4.1 Metrics Table
| Model            | MAE       | RMSE       | R²    |
|------------------|-----------|-----------|-------|
| **XGBoost**      | 114,999   | 237,510   | 0.9615 |
| RandomForest     | 134,603   | 261,416   | 0.9534 |
| GradientBoosting | 144,290   | 296,728   | 0.9399 |
| Ridge            | 168,472   | 337,496   | 0.9223 |
| ExtraTrees       | 165,253   | 385,512   | 0.8986 |
| Lasso            | 225,953   | 536,036   | 0.8039 |
| CatBoost         | 301,929   | 587,634   | 0.7643 |
| HistGB           | 578,334   | 779,126   | 0.5857 |
| LightGBM         | 574,109   | 790,132   | 0.5739 |
| MLP              | 661,712   | 1,379,416 | -0.2987 |

---


## 4.2 Why the Top 2 Models Perform Best

### 4.2.1 XGBoost — Why it’s #1 here
**Summary:**  
XGBoost tops the leaderboard (R² **0.9615**) because it captures **non‑linear interactions** among count‑style features (e.g., status pivots, `Total_Claims`) and a categorical priority signal, while controlling variance through regularization and tree‑wise subsampling.

**Reasons it fits this dataset:**
- **Handles tabular, mixed-scale features extremely well.**  
  Our features are mostly small‑integer counts (per‑status claim tallies) plus an encoded `PriorityID`. Gradient boosting trees naturally model non-linear thresholds and interactions (e.g., *“high Total_Claims + many PEND + Priority=High”*).
- **Strong bias–variance balance via regularization.**  
  Parameters like `learning_rate`, `max_depth`, `reg_alpha`, `reg_lambda`, and `subsample/colsample_bytree` help avoid overfitting while still learning complex structure.
- **Robust to outliers in the target.**  
  Boosted trees can localize high‑duration pockets (heavy tails common in process times) without distorting the rest of the fit, improving RMSE.
- **Additive improvements through boosting.**  
  Iteratively correcting residuals is effective when simple splits explain part of the variance (e.g., status mix), but nuanced interactions remain.
- **Efficient training and early stopping.**  
  XGBoost’s histogram-based tree construction and early stopping (if enabled) quickly converge to a strong solution on moderate-sized tabular data.

**Observed outcomes in our run:**
- **Lowest error:** MAE **114,999 sec**; RMSE **237,510 sec**  
- **Best generalization:** Highest R² (**0.9615**), beating RandomForest by a **9.15% RMSE improvement** and **14.56% MAE improvement**

---

### 4.2.2 RandomForest — Why it’s a strong runner‑up
**Summary:**  
RandomForest performs consistently well (R² **0.9534**) thanks to **bagging** and **feature randomness** that reduce variance and make it robust to noise and modest feature correlations.

**Reasons it fits this dataset:**
- **Excellent default for heterogeneous tabular features.**  
  Each tree handles discrete thresholds in count features and the encoded priority without requiring scaling or complex preprocessing.
- **Variance reduction through bagging.**  
  Bootstrapping rows + random feature subsets yield stable predictions even if some statuses are sparse or noisy over the last 10 days.
- **Resilient to overfitting with limited tuning.**  
  Compared with gradient boosting, RandomForest often succeeds with light hyperparameter work, which matches our modest grid.
- **Interpretability via feature importance.**  
  While not as granular as SHAP, impurity/perm-based importances give operations a quick read on influential signals (e.g., `Total_Claims`, key status counts).

**Observed outcomes in our run:**
- **Second-best across all metrics:** MAE **134,603 sec**; RMSE **261,416 sec**; R² **0.9534**  
- Small gap to XGBoost suggests most structure is captured by tree splits; the extra boost iterations in XGBoost squeeze out the last mile of accuracy.



## 4.3 Understanding Evaluation Metrics

### ✅ R² (Coefficient of Determination)
- **What it measures:**  
  How well the model explains the variance in the target variable (`Total_Time_Seconds`).
- **Interpretation:**  
  - R² = 1 → Perfect prediction (model explains 100% of variance).
  - R² = 0 → Model is no better than predicting the mean.
  - R² < 0 → Model performs worse than a simple mean predictor.
- **In our results:**  
  - XGBoost: **0.9615** → Explains ~96% of variance (excellent fit).
  - RandomForest: **0.9534** → Explains ~95% of variance.
  - MLP: **-0.2987** → Worse than predicting the average (poor fit).

---

### ✅ RMSE (Root Mean Squared Error)
- **What it measures:**  
  The square root of the average squared difference between predicted and actual values.
- **Interpretation:**  
  - Penalizes **large errors more heavily** because of squaring.
  - Lower RMSE = better accuracy.
- **Units:** Same as the target (seconds).
- **In our results:**  
  - XGBoost: **237,510 sec (~66 hours)** average error magnitude.
  - RandomForest: **261,416 sec (~72 hours)**.
  - MLP: **1,379,416 sec (~383 hours)** → Very poor.

---

### ✅ MAE (Mean Absolute Error)
- **What it measures:**  
  The average absolute difference between predicted and actual values.
- **Interpretation:**  
  - Easier to interpret than RMSE because it’s a straight average.
  - Less sensitive to outliers than RMSE.
- **Units:** Same as the target (seconds).
- **In our results:**  
  - XGBoost: **114,999 sec (~32 hours)** average deviation.
  - RandomForest: **134,603 sec (~37 hours)**.
  - MLP: **661,712 sec (~184 hours)** → Very poor.

---

### ✅ Summary Table
| Metric | What It Tells You |
|--------|--------------------|
| **R²** | How much variance in `Total_Time_Seconds` is explained by the model |
| **RMSE** | Typical size of prediction error (penalizes big mistakes) |
| **MAE** | Average prediction error (straightforward measure) |

---

**Why use all three?**
- R² shows **overall explanatory power**.
- RMSE and MAE show **error magnitude** (RMSE emphasizes big errors, MAE gives a balanced view).


### 4.4 Ranking Diagram
```markdown
![Performace Graph](./model_performance.png)
```
## 5. Artifacts Produced
- `aggregated_data.csv`
- `model_performance_results.csv`
- `performance_summary.csv`
- `predictions_with_original_features.csv`
- `model_performance.png`
- `best_model.pkl`

---
