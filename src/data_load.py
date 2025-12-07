import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_synthetic_health_data(n=800, random_state=42):
    np.random.seed(random_state)

    age = np.random.normal(50, 12, n).clip(18, 90)
    bmi = np.random.normal(27, 4, n).clip(16, 45)
    systolic_bp = np.random.normal(125, 15, n).clip(90, 200)
    cholesterol = np.random.normal(190, 35, n).clip(100, 350)
    glucose = np.random.normal(100, 25, n).clip(60, 250)
    smoker = np.random.binomial(1, 0.25, n)
    activity_level = np.random.choice(["low", "medium", "high"], size=n, p=[0.3, 0.5, 0.2])

    # Nonlinear interaction + small noise
    risk_score = (
        0.02 * (age - 50)
        + 0.04 * (bmi - 25)
        + 0.03 * (systolic_bp - 120)
        + 0.03 * (cholesterol - 180)
        + 0.02 * (glucose - 100)
        + 0.25 * smoker
        - 0.15 * (activity_level == "high").astype(int)
    )
    # interaction: high bmi amplifies glucose effect
    risk_score += 0.01 * ((bmi - 25) * (glucose - 100))

    risk_prob = 1 / (1 + np.exp(-risk_score))
    # set threshold via quantile to produce roughly balanced classes
    threshold = np.quantile(risk_prob, 0.55)
    risk = (risk_prob > threshold).astype(int)

    df = pd.DataFrame({
        "age": np.round(age, 2),
        "bmi": np.round(bmi, 2),
        "systolic_bp": np.round(systolic_bp, 2),
        "cholesterol": np.round(cholesterol, 2),
        "glucose": np.round(glucose, 2),
        "smoker": smoker,
        "activity_level": activity_level,
        "risk": risk
    })
    return df

def load_data(test_size=0.2, random_state=42):
    df = generate_synthetic_health_data(n=800, random_state=random_state)
    X = df.drop(columns=["risk"]).reset_index(drop=True)
    y = df["risk"].reset_index(drop=True)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

if __name__ == "__main__":
    df = generate_synthetic_health_data(n=10)
    print(df.head())