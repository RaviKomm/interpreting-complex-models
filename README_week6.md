
# Week 6 Assignment — Interpreting Complex Models in Healthcare

## Overview
This project demonstrates how to train a complex machine learning model on a synthetic healthcare dataset and apply interpretability techniques such as permutation importance, Partial Dependence Plots (PDP), and Individual Conditional Expectation (ICE).

The assignment follows the required workflow using:
- Python scripts (no notebooks)
- VS Code
- GitHub version control
- scikit-learn, NumPy, Pandas, Matplotlib

---

## Dataset
A **custom synthetic healthcare dataset** was generated using Python.  
It contains 600 samples with features resembling real clinical predictors:

**Numerical Features**
- Age  
- BMI  
- Systolic blood pressure  
- Cholesterol  
- Glucose  

**Categorical Features**
- Activity level  
- Smoker status  

**Target**
- `risk`: Binary classification (0 = low risk, 1 = high risk)

---

## Project Structure
```
/src
    data_load.py
    model_train.py
    evaluate.py
    interpret.py
    main.py
/models
    model.joblib
/figures
    roc_curve.png
    permutation_importance.png
    pdp_ice_age.png
    pdp_ice_bmi.png
    pdp_ice_glucose.png

README.md
requirements.txt
```

---

## How to Run

### 1. Create virtual environment
```
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate    # Windows
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run the full pipeline
```
python src/main.py
```

This will:
- Generate synthetic data  
- Train a Gradient Boosting model  
- Evaluate on test set  
- Produce interpretability figures  
- Save everything in `models/` and `figures/`

---

## Results Summary

### **Model Performance**
(Values may vary slightly)
- Accuracy: ~0.93  
- Precision: ~0.91  
- Recall: ~0.94  
- F1-score: ~0.92  
- ROC-AUC: ~0.97  

### **Interpretability Outputs**
Produced in the `/figures` folder:
- ROC curve  
- Permutation importance  
- PDP + ICE for:
  - Age  
  - BMI  
  - Glucose  

---

## Key Insights for Clinicians
- **Risk increases sharply when glucose exceeds ~150 mg/dL.**
- **High BMI and low physical activity jointly increase risk.**
- **Age shows a monotonic upward trend with predicted risk.**
- **Permutation importance confirms glucose, BMI, and blood pressure as high-impact factors.**

---

## Limitations
- Synthetic dataset—real-world biases are not present  
- Correlated features may distort PDP interpretations  
- Model probabilities may need calibration  
- Clinical validation required for real deployment  

---

## References (APA)
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- scikit-learn documentation: https://scikit-learn.org/
