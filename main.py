import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# Crear carpeta de salida si no existe
os.makedirs('output', exist_ok=True)

# Leer el archivo CSV, saltando la primera fila
df = pd.read_csv("data/ganancias.csv", skiprows=1)

# Verificar columnas
print("Columnas detectadas:", df.columns)

# Transformar de formato ancho a largo
df_largo = df.melt(id_vars="mes", var_name="año", value_name="ganancias")

# Orden correcto de meses
orden_meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio"]
df_largo["mes"] = pd.Categorical(df_largo["mes"], categories=orden_meses, ordered=True)

# Gráfico de líneas
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_largo, x="mes", y="ganancias", hue="año", marker="o")
plt.title("Ganancias del 1er Semestre (2021–2023)")
plt.xlabel("Mes")
plt.ylabel("Ganancias")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/lineas.png", dpi=300)
plt.close()

# Clustering KMeans
df_cluster = df_largo.groupby(["año", "mes"], as_index=False, observed=True)["ganancias"].sum()
X = df_cluster[["ganancias"]]
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster["cluster"] = kmeans.fit_predict(X)

plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_cluster, x="mes", y="ganancias", hue="cluster", palette="Set1")
plt.title("Clustering de Ganancias por Mes (KMeans)")
plt.xlabel("Mes")
plt.ylabel("Ganancias")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/kmeans_clustering.png", dpi=300)
plt.close()

# Gráfico de Barras – Totales por Año
totales_por_año = df_largo.groupby("año")["ganancias"].sum().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(data=totales_por_año, x="año", y="ganancias", palette="pastel")
plt.title("Ganancias Totales por Año")
plt.xlabel("Año")
plt.ylabel("Ganancia Total")
plt.tight_layout()
plt.savefig("output/barras_totales_por_año.png", dpi=300)
plt.close()

# Heatmap – Ganancias por Mes y Año
pivot = df_largo.pivot(index="mes", columns="año", values="ganancias")
python main.py
plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Ganancias por Mes y Año (Heatmap)")
plt.xlabel("Año")
plt.ylabel("Mes")
plt.tight_layout()
plt.savefig("output/heatmap_ganancias.png", dpi=300)
plt.close()

# Boxplot – Distribución de Ganancias por Año
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_largo, x="año", y="ganancias", palette="Set2")
plt.title("Distribución de Ganancias por Año")
plt.xlabel("Año")
plt.ylabel("Ganancia")
plt.tight_layout()
plt.savefig("output/boxplot_ganancias.png", dpi=300)
plt.close()

# Regresión Lineal – Predicción
df_mean = df_largo.groupby("año")["ganancias"].mean().reset_index()
df_mean["año_num"] = df_mean["año"].astype(int)
X = df_mean[["año_num"]]
y = df_mean["ganancias"]
modelo = LinearRegression()
modelo.fit(X, y)
y_pred = modelo.predict(X)

plt.figure(figsize=(8, 5))
sns.scatterplot(x="año_num", y="ganancias", data=df_mean, label="Ganancia promedio")
plt.plot(df_mean["año_num"], y_pred, color="red", linestyle="--", label="Tendencia")
plt.title("Regresión Lineal – Ganancia Promedio por Año")
plt.xlabel("Año")
plt.ylabel("Ganancia Promedio")
plt.legend()
plt.tight_layout()
plt.savefig("output/regresion_ganancias.png", dpi=300)
plt.close()
