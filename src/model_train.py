"""
model_train.py
Builds a Pipeline (ColumnTransformer -> HistGradientBoostingClassifier), fits on train,
and saves the trained pipeline to disk.
"""
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

def build_pipeline(numeric_features, categorical_features, random_state=42):
    # numeric transformer
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    # categorical transformer (one-hot)
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    clf = HistGradientBoostingClassifier(random_state=random_state)
    pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
    return pipe

def train_and_save(X_train, y_train, numeric_features, categorical_features,
                   model_path='models/model.joblib', random_state=42):
    pipe = build_pipeline(numeric_features, categorical_features, random_state=random_state)
    pipe.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipe, model_path)
    print(f"Saved model to {model_path}")
    return pipe
