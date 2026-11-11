"""Fase 1: ingesta, diagnóstico de overfitting, Random Forest optimizado y serialización."""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             roc_curve, roc_auc_score)
from src.database.connection2 import MySQLConnector

QUERY = """
    SELECT p.player_id, p.name AS jugador, p.position AS posicion,
           p.height_in_cm AS estatura_cm,
           TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
           p.market_value_in_eur AS valor_mercado
    FROM players p
    WHERE p.market_value_in_eur IS NOT NULL AND p.date_of_birth IS NOT NULL;
"""
RUTA_MODELO = "models/modelo_scouting_rf.joblib"
CARPETA_REPORTES = "reports"


def construir_X(d, columnas):
    """Matriz X alineada con las columnas vistas en entrenamiento."""
    dummies = pd.get_dummies(d["posicion"], prefix="posicion", dtype=int)
    X = pd.concat([d[["edad", "estatura_cm"]], dummies], axis=1)
    return X.reindex(columns=columnas, fill_value=0)


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(CARPETA_REPORTES, exist_ok=True)

    # ---------- 1. INGESTA Y PREPROCESAMIENTO ----------
    df = MySQLConnector().extract_data(QUERY)
    for c in ["estatura_cm", "edad", "valor_mercado"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["posicion"] = df["posicion"].fillna("Sin Posicion")

    # Target binario: Top 25 % del valor de mercado
    umbral_elite = df["valor_mercado"].quantile(0.75)
    df["es_elite"] = (df["valor_mercado"] >= umbral_elite).astype(int)

    # Split estratificado 80/20 (antes de calcular medianas, para evitar fuga de datos)
    df_tr, df_te = train_test_split(df, test_size=0.20, random_state=42, stratify=df["es_elite"])

    # Imputación por mediana agrupada, aprendida SOLO con Train
    medianas_estatura = df_tr.groupby("posicion")["estatura_cm"].median().dropna().to_dict()
    mediana_global_estatura = df_tr["estatura_cm"].median()
    mediana_global_edad = df_tr["edad"].median()

    def imputar(d):
        d = d.copy()
        d["estatura_cm"] = (d["estatura_cm"].fillna(d["posicion"].map(medianas_estatura))
                            .fillna(mediana_global_estatura))
        d["edad"] = d["edad"].fillna(mediana_global_edad)
        return d

    df_tr, df_te = imputar(df_tr), imputar(df_te)

    # One-Hot Encoding (drop_first en Train; Test se alinea a esas columnas)
    dum_tr = pd.get_dummies(df_tr["posicion"], prefix="posicion", drop_first=True, dtype=int)
    X_train = pd.concat([df_tr[["edad", "estatura_cm"]], dum_tr], axis=1)
    columnas = X_train.columns.tolist()
    X_test = construir_X(df_te, columnas)
    y_train, y_test = df_tr["es_elite"], df_te["es_elite"]
    print(f"[OK] Train: {len(X_train)} | Test: {len(X_test)} | "
          f"Élite en Test: {y_test.mean():.2%} | Umbral Élite: €{umbral_elite:,.0f}")

    # ---------- 2. DIAGNÓSTICO DE OVERFITTING ----------
    libre = DecisionTreeClassifier(max_depth=None, random_state=42).fit(X_train, y_train)
    podado = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42).fit(X_train, y_train)
    filas = []
    for nombre, m in [("Árbol sin restricción", libre), ("Árbol con poda previa", podado)]:
        a_tr = accuracy_score(y_train, m.predict(X_train))
        a_te = accuracy_score(y_test, m.predict(X_test))
        filas.append({"Modelo": nombre, "Accuracy Train": a_tr, "Accuracy Test": a_te,
                      "Brecha": a_tr - a_te, "F1 Élite Test": f1_score(y_test, m.predict(X_test), zero_division=0)})
    tabla = pd.DataFrame(filas).set_index("Modelo")
    print("\n=== DIAGNÓSTICO DE OVERFITTING ===\n", tabla.round(4).to_string())
    tabla.round(4).to_csv(f"{CARPETA_REPORTES}/tabla_overfitting.csv")

    ax = tabla[["Accuracy Train", "Accuracy Test"]].plot(kind="bar", rot=0, figsize=(7, 4.5), edgecolor="black")
    ax.set_ylabel("Accuracy"); ax.set_title("Train vs Test: árbol libre vs árbol podado")
    ax.set_ylim(0, 1.05)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{CARPETA_REPORTES}/overfitting_train_vs_test.png", dpi=200); plt.close()

    # ---------- 3. RANDOM FOREST + GRIDSEARCHCV ----------
    grid = {"n_estimators": [100, 200], "max_depth": [4, 6, 8],
            "min_samples_leaf": [5, 10], "class_weight": ["balanced", None]}
    gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), grid,
                      cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    rf = gs.best_estimator_
    print(f"\n[GridSearchCV] Mejores parámetros: {gs.best_params_} | F1 CV: {gs.best_score_:.4f}")
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Estándar", "Élite"], zero_division=0))

    # ---------- 4. CURVA ROC Y AUC ----------
    plt.figure(figsize=(6, 5))
    aucs = {}
    for nombre, m in [("Árbol podado", podado), ("Random Forest", rf)]:
        p = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        aucs[nombre] = roc_auc_score(y_test, p)
        plt.plot(fpr, tpr, lw=2, label=f"{nombre} (AUC = {aucs[nombre]:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Azar (AUC = 0.5)")
    plt.xlabel("Tasa de falsos positivos (FPR)"); plt.ylabel("Tasa de verdaderos positivos (TPR)")
    plt.title("Curvas ROC: árbol único vs Random Forest"); plt.legend(loc="lower right"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{CARPETA_REPORTES}/roc_arbol_vs_rf.png", dpi=200); plt.close()
    print("AUC:", {k: round(v, 4) for k, v in aucs.items()})

    # ---------- 5. IMPORTANCIA DE VARIABLES (GINI) ----------
    imp = pd.Series(rf.feature_importances_, index=columnas).sort_values()
    plt.figure(figsize=(7, 4.5)); imp.plot(kind="barh", edgecolor="black")
    plt.xlabel("Importancia de Gini"); plt.title("Importancia de variables - Random Forest")
    plt.tight_layout(); plt.savefig(f"{CARPETA_REPORTES}/importancia_variables.png", dpi=200); plt.close()
    print("\n=== IMPORTANCIA DE VARIABLES ===\n", imp.sort_values(ascending=False).round(4).to_string())

    # ---------- 6. SERIALIZACIÓN ----------
    joblib.dump({
        "modelo": rf,
        "columnas_X": columnas,
        "medianas_estatura": medianas_estatura,
        "mediana_global_estatura": float(mediana_global_estatura),
        "mediana_global_edad": float(mediana_global_edad),
        "umbral_elite_eur": float(umbral_elite),
        "auc_test": float(aucs["Random Forest"]),
    }, RUTA_MODELO)
    print(f"\n[OK] Paquete exportado en '{RUTA_MODELO}'")


if __name__ == "__main__":
    main()
