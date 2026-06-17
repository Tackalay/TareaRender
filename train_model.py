import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Student_Performance.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_student_performance.joblib"
METRICS_PATH = BASE_DIR / "models" / "metricas.json"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Performance Index"])
    y = df["Performance Index"]

    columnas_numericas = [
        "Hours Studied",
        "Previous Scores",
        "Sleep Hours",
        "Sample Question Papers Practiced",
    ]
    columnas_categoricas = ["Extracurricular Activities"]

    preprocesamiento = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), columnas_categoricas),
        ("num", "passthrough", columnas_numericas),
    ])

    modelo = Pipeline([
        ("preprocess", preprocesamiento),
        ("model", LinearRegression()),
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores_r2 = cross_val_score(modelo, X, y, cv=cv, scoring="r2")
    scores_mae = -cross_val_score(modelo, X, y, cv=cv, scoring="neg_mean_absolute_error")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    metricas = {
        "r2_cv_mean": float(scores_r2.mean()),
        "r2_cv_std": float(scores_r2.std()),
        "mae_cv_mean": float(scores_mae.mean()),
        "mae_cv_std": float(scores_mae.std()),
        "r2_extra": float(r2_score(y_test, y_pred)),
        "mae_extra": float(mean_absolute_error(y_test, y_pred)),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "features": X.columns.tolist(),
        "target": "Performance Index",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metricas, file, ensure_ascii=False, indent=2)

    print("Modelo entrenado correctamente.")
    print(json.dumps(metricas, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
