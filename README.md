# Predicción de Rendimiento Académico

Aplicación web en Flask para predecir el `Performance Index` de un estudiante usando un modelo de regresión lineal entrenado con el dataset `Student_Performance.csv`.

## Variables de entrada

- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced

## Variable objetivo

- Performance Index

## Métricas del modelo

- R² promedio CV=5: 0.9887
- MAE promedio CV=5: 1.6186
- R² corrida independiente 80/20: 0.9890
- MAE corrida independiente 80/20: 1.6111

## Ejecutar localmente

```bash
pip install -r requirements.txt
python app.py
```

Después abre:

```text
http://127.0.0.1:5000
```

## Subir a Render

1. Subir este proyecto a un repositorio de GitHub.
2. Entrar a Render y crear un nuevo Web Service.
3. Conectar el repositorio de GitHub.
4. Configurar:
   - Language: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
5. Crear el servicio y esperar el despliegue.
