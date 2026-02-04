# Отличный набор заданий — это прямо **мини-production ML-система** 👍
# Покажу **единый шаблон проекта + заготовки кода** для каждого пункта. Без лишней магии, чтобы ты понимал, *что за что отвечает* и мог спокойно дорабатывать.
#
# ---
#
# # 📁 Общая структура проекта (рекомендую)
#
# ```
# project/
# │
# ├── data/
# │   ├── raw/
# │   ├── processed/
# │   └── new_data/
# │
# ├── models/
# │   ├── versions/
# │   └── best_model.pkl
# │
# ├── logs/
# │   └── training.log
# │
# ├── src/
# │   ├── training/
# │   │   ├── train.py
# │   │   ├── evaluate.py
# │   │   └── drift.py
# │   │
# │   ├── inference/
# │   │   └── predict.py
# │   │
# │   ├── api/
# │   │   └── app.py
# │   │
# │   └── utils/
# │       ├── io.py
# │       └── metrics.py
# │
# ├── ui/
# │   └── app.py
# │
# ├── requirements.txt
# └── README.md
# ```
#
# ---
#
# # 🔹 3.1 Обучение модели
#
# ## `train.py` — агент обучения
#
# ```python
# import joblib
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, roc_auc_score
#
# MODELS = {
#     "logreg": LogisticRegression(max_iter=1000),
#     "rf": RandomForestClassifier(n_estimators=200, random_state=42)
# }
#
# def train_models(X, y):
#     results = {}
#
#     for name, model in MODELS.items():
#         model.fit(X, y)
#         preds = model.predict(X)
#         proba = model.predict_proba(X)
#
#         results[name] = {
#             "model": model,
#             "report": classification_report(y, preds, output_dict=True),
#             "roc_auc": roc_auc_score(y, proba, multi_class="ovr")
#         }
#
#     return results
# ```
#
# ---
#
# ## `evaluate.py` — выбор лучшей модели
#
# ```python
# def select_best_model(results):
#     best_name = max(results, key=lambda x: results[x]["roc_auc"])
#     return best_name, results[best_name]["model"]
# ```
#
# ---
#
# ## `main` обучение + сохранение
#
# ```python
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# results = train_models(X_train, y_train)
# best_name, best_model = select_best_model(results)
#
# joblib.dump(best_model, f"models/best_model.pkl")
# ```
#
# ---
#
# # 🔹 3.2 Непрерывное обучение и контроль дрейфа
#
# ## `drift.py` — проверка дрейфа (просто и понятно)
#
# ```python
# import numpy as np
# from scipy.stats import ks_2samp
#
# def check_drift(old_data, new_data, threshold=0.05):
#     drift_features = []
#
#     for col in old_data.columns:
#         stat, p_value = ks_2samp(old_data[col], new_data[col])
#         if p_value < threshold:
#             drift_features.append(col)
#
#     return drift_features
# ```
#
# ---
#
# ## Логика дообучения
#
# ```python
# if drift_detected:
#     retrain_from_scratch()
# else:
#     model.fit(new_X, new_y)
# ```
#
# ---
#
# ## Версионирование
#
# ```python
# from datetime import datetime
#
# version = datetime.now().strftime("%Y%m%d_%H%M")
# joblib.dump(model, f"models/versions/model_{version}.pkl")
# ```
#
# ---
#
# # 🔹 4.1 API (FastAPI)
#
# ## `app.py`
#
# ```python
# from fastapi import FastAPI
# import joblib
# import pandas as pd
#
# app = FastAPI()
# model = joblib.load("models/best_model.pkl")
#
# @app.post("/predict_risk")
# def predict_risk(data: list[dict]):
#     df = pd.DataFrame(data)
#     preds = model.predict(df)
#
#     return {"risk_level": preds.tolist()}
# ```
#
# ---
#
# ## Прогноз объёма транзакций (заглушка)
#
# ```python
# @app.get("/forecast_volume")
# def forecast_volume(months: int = 3):
#     return {
#         "months": months,
#         "forecast": [10000, 12000, 15000]
#     }
# ```
#
# ---
#
# # 🔹 4.2 GUI (Streamlit — идеально для этого)
#
# ## `ui/app.py`
#
# ```python
# import streamlit as st
# import pandas as pd
# import requests
#
# st.title("Оценка риска транзакций")
#
# uploaded_file = st.file_uploader("Загрузите CSV")
#
# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     response = requests.post(
#         "http://localhost:8000/predict_risk",
#         json=df.to_dict(orient="records")
#     )
#     df["risk"] = response.json()["risk_level"]
#     st.dataframe(df)
#
# if st.button("Справка"):
#     st.info("""
#     Уровни риска:
#     - Низкий
#     - Средний
#     - Высокий
#     """)
# ```
#
# ---
#
# # 🧠 Как это защищать / объяснять
#
# * **3.1** — агент выбирает модель по объективным метрикам
# * **3.2** — автоматизация жизненного цикла модели (MLOps-подход)
# * **4.1** — API изолировано от обучения → безопасно
# * **4.2** — интерфейс для не-технических пользователей
#
# ---
#
# Если хочешь, дальше можем:
#
# * ✍️ оформить это как **курсовую / диплом**
# * 📊 добавить **SHAP-интерпретацию**
# * 🧪 прикрутить **pytest**
# * 🧱 довести до уровня **real production**
#
# Скажи, что из этого тебе сейчас важнее 👌
