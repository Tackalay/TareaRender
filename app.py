from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import json
from io import BytesIO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo_student_performance.joblib"
METRICS_PATH = BASE_DIR / "models" / "metricas.json"
DATA_PATH = BASE_DIR / "data" / "Student_Performance.csv"

app = Flask(__name__)

modelo = joblib.load(MODEL_PATH)
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metricas = json.load(f)


def crear_dataframe_entrada(formulario):
    """Convierte los datos del formulario en el formato esperado por el modelo."""
    return pd.DataFrame([{
        "Hours Studied": float(formulario["hours_studied"]),
        "Previous Scores": float(formulario["previous_scores"]),
        "Extracurricular Activities": formulario["extracurricular_activities"],
        "Sleep Hours": float(formulario["sleep_hours"]),
        "Sample Question Papers Practiced": float(formulario["sample_papers"]),
    }])


@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    error = None
    valores = {
        "hours_studied": "",
        "previous_scores": "",
        "extracurricular_activities": "Yes",
        "sleep_hours": "",
        "sample_papers": "",
    }

    if request.method == "POST":
        valores = request.form.to_dict()
        try:
            datos = crear_dataframe_entrada(request.form)
            resultado = float(modelo.predict(datos)[0])
            # El dataset maneja un índice de desempeño de 10 a 100.
            prediccion = max(10, min(100, resultado))
        except Exception as exc:
            error = f"No se pudo realizar la predicción. Revisa los datos ingresados. Detalle: {exc}"

    return render_template(
        "index.html",
        prediccion=prediccion,
        error=error,
        valores=valores,
        metricas=metricas,
    )


@app.route("/download/csv")
def descargar_csv():
    return send_file(DATA_PATH, as_attachment=True, download_name="Student_Performance.csv")


@app.route("/download/excel")
def descargar_excel():
    df = pd.read_csv(DATA_PATH)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Student Performance")
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="Student_Performance.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    app.run(debug=True)
