import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse
from typing import Dict
import json
import uvicorn
from src.model.clasificador import *

class UserInput(BaseModel):
    mensaje_usuario: str = Field(..., example="mensaje", description="Necesidad del usuario")

class ChatResponse(BaseModel):
    response: str  # La respuesta generada por el modelo LLM

# Crear la aplicación FastAPI
app = FastAPI()

# Configuración de CORS para permitir solicitudes desde cualquier origen
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoint para manejar el chat del usuario
# Endpoint para manejar el chat del usuario
@app.post("/chat", response_model=ChatResponse)
async def chat(user_input: UserInput):
    
    respuesta = clasificador_rag(user_input.mensaje_usuario)
    
    if not respuesta:
        return ChatResponse(response="No se recibió una respuesta del modelo")
    
    # Dado que respuesta ya es un dict, no es necesario json.loads
    response_text = respuesta.get("respuesta", "Lo siento, no pude entender la pregunta. Por favor, intenta nuevamente.")

    return ChatResponse(response=response_text)
    
    
# Endpoint para redirigir a la documentación interactiva de FastAPI
@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# Ejecutar el servidor de desarrollo de uvicorn
if __name__ == '__main__':
    puerto = 8080
    uvicorn.run(app, host="0.0.0.0", port=int(puerto))  # Escuchar en todas las interfaces de red



# docker build -t my-app .
# docker run -d -p 8000:8000 my-app
# docker save -o myapp.tar my-app
