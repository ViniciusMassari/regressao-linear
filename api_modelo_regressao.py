from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criar uma instância do FastApi

app = FastAPI()

# Criar uma que terá os dados do request body para a API
class RequestBody(BaseModel):
    horas_estudo: float


# Carregar modelo para realizar a predição
modelo_reg_pontuacao = joblib.load("./modelo_regressao.pkl")

@app.post("/predict")
def predict(data: RequestBody):
    # Preparar os dados para a predição
    input_feature = [[data.horas_estudo]]

    # Realizar a predição
    y_pred = modelo_reg_pontuacao.predict(input_feature)[0].astype(int)

    return {
        "pontuacao_teste": y_pred.tolist()
    }