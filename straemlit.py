# # app.py
# import streamlit as st
# import pandas as pd
# import sqlite3
# import plotly.express as px
#
# # Подключаемся к базе
# conn = sqlite3.connect("data.db")
#
# # Загружаем таблицу
# df = pd.read_sql_query("SELECT * FROM sales", conn)
#
# st.title("📊 Аналитический дэшборд по продажам")
#
# # Фильтр по году
# year = st.selectbox("Выбери год", sorted(df["year"].unique()))
# filtered = df[df["year"] == year]
#
# # График
# fig = px.bar(filtered, x="month", y="revenue", title="Доход по месяцам")
# st.plotly_chart(fig)
#
# # Таблица
# st.dataframe(filtered)
#
# conn.close()
