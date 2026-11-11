"""Fase 2: API REST con FastAPI. Ejecutar: uvicorn main_api:app --reload"""
import os
from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

RUTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "modelo_scouting_rf.joblib")
paquete = joblib.load(RUTA)
modelo = paquete["modelo"]
columnas = paquete["columnas_X"]

app = FastAPI(title="API de Scouting Deportivo", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Jugador(BaseModel):
    jugador: Optional[str] = None
    posicion: Optional[str] = None
    edad: Optional[float] = Field(None, ge=14, le=50)
    estatura_cm: Optional[float] = Field(None, ge=140, le=220)


@app.get("/")
def estado():
    return {"estado": "activo", "umbral_elite_eur": paquete["umbral_elite_eur"]}


@app.post("/predict")
def predict(j: Jugador):
    # Mismas reglas de imputación aprendidas en entrenamiento
    pos = (j.posicion or "").strip() or "Sin Posicion"
    estatura = j.estatura_cm if j.estatura_cm is not None else \
        paquete["medianas_estatura"].get(pos, paquete["mediana_global_estatura"])
    edad = j.edad if j.edad is not None else paquete["mediana_global_edad"]

    valores = {c: 0.0 for c in columnas}
    valores["edad"] = float(edad)
    valores["estatura_cm"] = float(estatura)
    if f"posicion_{pos}" in valores:
        valores[f"posicion_{pos}"] = 1.0
    fila = pd.DataFrame([valores], columns=columnas)

    es_elite = int(modelo.predict(fila)[0])
    prob = float(modelo.predict_proba(fila)[0, 1])
    return {
        "jugador": j.jugador,
        "posicion": pos,
        "edad": round(float(edad), 1),
        "estatura_cm": round(float(estatura), 1),
        "prediccion": "Élite" if es_elite else "Estándar",
        "probabilidad_elite": round(prob * 100, 2),
    }
