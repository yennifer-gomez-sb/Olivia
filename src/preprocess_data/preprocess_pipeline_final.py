import pandas as pd
import json
import os
import re
from langchain_openai import AzureChatOpenAI

# Configuración de variables de entorno
response = """
AZURE_OPENAI_API_KEY=bbead62b15004f6f901f5563527c855b
OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_ENDPOINT=https://ai-cc-desa.openai.azure.com/
GPT4_NAME=ANALITICA-GPT4
GPT35_NAME=ANALITICA-GPT3516K
EMBEDDINGS_NAME=ANALITICA-ADA002
WHISPER_NAME=ANALITICA-WHISPER
"""

keys = response.splitlines()
for key in keys:
    if key:  # Evitar líneas vacías
        name, value = key.split("=")
        os.environ[name] = value

# Modelo LLM GPT 4 usando Azure OpenAI
llm_model = AzureChatOpenAI(
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment=os.getenv("GPT4_NAME")
)

# Función principal

def procesar_datos(ruta_excel, hoja, llm_model):
    # Leer el archivo inicial
    df = pd.read_excel(ruta_excel, sheet_name=hoja)
    nuevas_filas = []

    # Separar preguntas y respuestas
    for index, row in df.iterrows():
        contenido = str(row['Content'])
        metadata = row.get('Metadata', '')

        pregunta_match = re.search(r'PREGUNTA:(.*?)RESPUESTA:', contenido, re.DOTALL)
        respuesta_match = re.search(r'RESPUESTA:(.*)', contenido, re.DOTALL)

        if pregunta_match and respuesta_match:
            pregunta = pregunta_match.group(1).strip()
            respuesta = respuesta_match.group(1).strip()

            nuevas_filas.append({'Tipo': 'PREGUNTA', 'Contenido': pregunta, 'Metadata': metadata})
            nuevas_filas.append({'Tipo': 'RESPUESTA', 'Contenido': respuesta, 'Metadata': metadata})

    df_separado = pd.DataFrame(nuevas_filas)

    # Generar contenido multiturn
    textsi_1 = """
    Recibes como entrada varios grupos de preguntas, donde cada grupo comparte una misma respuesta.
    Tu tarea es generar un resumen claro y preciso para cada grupo de preguntas, capturando su significado completo
    y los matices esenciales que las conectan con su respuesta.
    """

    for index, row in df_separado[df_separado['Tipo'] == 'PREGUNTA'].iterrows():
        text_input = row['Contenido']
        try:
            response = llm_model.invoke(f"{textsi_1}\n\n{text_input}")
            df_separado.at[index, 'Contenido'] = response.content
        except Exception as e:
            print(f"Error en fila {index}: {e}")

    # Combinar preguntas y respuestas
    combinados = []

    for i in range(0, len(df_separado) - 1, 2):
        if df_separado.iloc[i]['Tipo'] == 'PREGUNTA' and df_separado.iloc[i + 1]['Tipo'] == 'RESPUESTA':
            pregunta = re.sub(r'^\*\*Resumen:\*\*\s*', '', df_separado.iloc[i]['Contenido']).strip()
            respuesta = df_separado.iloc[i + 1]['Contenido']
            metadata = df_separado.iloc[i]['Metadata']
            combinados.append({'Pregunta': pregunta, 'Respuesta': respuesta, 'Metadata': metadata})

    df_combinado = pd.DataFrame(combinados)

    # Crear DataFrames por metadata
    df_combinado['Metadata_dict'] = df_combinado['Metadata'].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
    
    df_sb = df_combinado[df_combinado['Metadata_dict'].apply(lambda x: "sb" in x.get('Dominio', []))]
    df_ac = df_combinado[df_combinado['Metadata_dict'].apply(lambda x: "ac" in x.get('Dominio', []))]

    df_sb.drop(columns=['Metadata_dict'], inplace=True)
    df_ac.drop(columns=['Metadata_dict'], inplace=True)

    # Guardar resultados finales
    df_sb.to_csv("data_sb_open.csv", index=False)
    df_ac.to_csv("data_ac_open.csv", index=False)

    return df_sb, df_ac

# Ejecutar el proceso completo
ruta_excel = "raw.xlsx"
hoja = "01082024"
df_sb, df_ac = procesar_datos(ruta_excel, hoja, llm_model)

print("¡Proceso completado con éxito! Archivos generados:")
print("- data_sb_open.xlsx")
print("- data_ac_open.xlsx")
