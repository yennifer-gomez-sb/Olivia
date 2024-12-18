import os
import chromadb
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAI
from google.cloud import secretmanager
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
import re
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import pandas as pd
import sys 

current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(os.path.dirname(current_dir)) 
sys.path.append(root_dir)

from utils.paths import embedding_model, llm_model_gemini_flash,client, collection_name_ac, collection_name_sb, llm_model_gemini, llm_model_gemini_pro, llm_model_gpt_4, llm_model_gpt_35
from utils.get_vectorstore import get_dataframe_from_chroma



collection_name = collection_name_ac

# Crear colección con la función de distancia de coseno
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"} # l2 is the default
)

# vector store
vector_store = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=embedding_model,
)


def format_docs(docs):
    # Ponerle a cada uno la metadata para que responda con los codigos de producto
    return "\n\n".join(f"""Pregunta: {doc.metadata['Pregunta']}
                    Respuesta: {doc.metadata['Respuesta']}"""
                    for doc in docs)


instruction_rag = """
Eres Olivia, un agente virtual experto en estructurar y validar preguntas frecuentes basadas en el material que se te proporciona.

Tu tarea es:
1. Revisar las preguntas proporcionadas para asegurarte de que son claras, completas y útiles.
2. Organizar las preguntas y respuestas en categorías o temas clave para facilitar la comprensión.
3. Proponer mejoras en las formulaciones de las preguntas si encuentras redundancias o poca claridad.
4. Validar que las respuestas sean precisas y estén alineadas con el contenido proporcionado.

El material que se te proporciona incluye el listado de preguntas y respuestas. Basándote en este material, organízalo y optimízalo para que sea fácil de entender y útil para los usuarios.

<OBJECTIVE_AND_PERSONA>
- Eres Olivia, un agente virtual de atención al cliente de Seguros Bolívar.
- Tu tarea es proporcionar respuestas precisas y útiles basándote únicamente en el contexto proporcionado, alineándote con los servicios y políticas de Seguros Bolívar.
</OBJECTIVE_AND_PERSONA>

<INSTRUCTIONS>
1. Analiza cuidadosamente la pregunta del cliente y el contexto proporcionado.
2. Genera una respuesta clara y profesional basada únicamente en la información del contexto:
   - **Coincidencia exacta**: Usa la información tal cual si responde directamente a la pregunta.
   - **Coincidencia parcial**: Construye una respuesta utilizando lo que esté disponible en el contexto.
   - **Sin información suficiente**: Si el contexto no incluye datos relevantes, responde con el mensaje de respuesta predeterminada (ver abajo).
3. Si el contexto incluye enlaces relevantes, incorpóralos en la respuesta.
4. Devuelve la respuesta en formato JSON usando únicamente el campo "respuesta".
</INSTRUCTIONS>

<CONSTRAINTS>

1. Dos:
- Usa un lenguaje profesional, claro y directo.
- Responde de forma breve y orientada a la acción.
- Asegúrate de que la respuesta sea completamente coherente con el contexto proporcionado.
2. Don´ts: 
- Evita dar información adicional al cliente.
- Evita mencionar explícitamente que falta información en el contexto. Usa la respuesta predeterminada.

</CONSTRAINTS>

6. **Formato de salida**:
    - Devuelve siempre las respuestas en formato JSON utilizando exclusivamente el campo "respuesta".

**Ejemplos de formato:**

Formato incorrecto:
    "Lamentablemente, no dispongo de información específica acerca de testimonios o reseñas de otros clientes en el contexto proporcionado."

Formato correcto:
    {{"respuesta": "¡Lo siento! Aún estoy aprendiendo y no cuento con la información que necesita. Por favor, escríbanos a nuestro WhatsApp dando clic aquí: 3223322322 o llámenos al #322 para resolver sus dudas."}}

### INPUT_FORMAT:


  "pregunta_cliente": "{question}",
  "contexto_disponible": "{context}"


<OUTPUT_FORMAT>

  "respuesta": "[Aquí debes incluir una respuesta clara, específica y basada en el contexto proporcionado. Si no se encuentra la respuesta, sigue esta plantilla: 'Lo siento, no encontré información específica sobre tu consulta. Por favor, comunícate con el soporte al #322 o utiliza el WhatsApp oficial: [3223322322](https://wa.link/2gsvmz).']",
  "fuente": "[Especifica si la respuesta se basó en el contexto, fue inferida o es una redirección al soporte.]"
</OUTPUT_FORMAT>

<RECAP>
Recuerda que debes actuar como Olivia, un agente virtual de Seguros Bolívar. Debes analizar la pregunta del cliente y el contexto proporcionado, generar una respuesta clara y profesional basada en la información del contexto, incluyendo enlaces relevantes si están disponibles. Si no hay información suficiente, utiliza la respuesta predeterminada. La respuesta debe ser en formato JSON, incluyendo los campos "respuesta" y "fuente". Evita dar información adicional que no esté en el contexto y no menciones explícitamente la falta de información.
</RECAP>

"""

def limpiar(texto):
    texto = re.sub(r"```json", "", texto)
    texto = re.sub(r"```", "", texto)
    
    # Elimina los \n que están justo después de '{' o justo antes de '}'
    texto = re.sub(r'\{\s*\\n', '{', texto)
    texto = re.sub(r'\\n\s*\}', '}', texto)
    
    # Elimina todo lo que esté antes de la primera llave {
    texto = re.sub(r'^.*?\{', '{', texto)
    # Elimina todo lo que esté después de la última llave }
    texto = re.sub(r'\}.*?$', '}', texto)
    
    return texto


# Definir la clase SimpleDocument que se utilizará para BM25Retriever
class SimpleDocument:
    def __init__(self, question, answer):
        # Almacenar tanto la pregunta como la respuesta en page_content
        self.page_content = f"Question: {question}\nAnswer: {answer}"
        # Mantener la metadata con 'Pregunta' y 'Respuesta'
        self.metadata = {
            'Pregunta': question,
            'Respuesta': answer
        }


df = get_dataframe_from_chroma(client, collection_name)

# Crear documentos utilizando SimpleDocument
docs = [SimpleDocument(row['Pregunta'], row['Respuesta']) for _, row in df.iterrows()]

# Crear el BM25 retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4



# RAG CHAIN  
def clasificador_rag(question):
    chroma_retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 4}
    )

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.35, 0.65])
    
    custom_rag_prompt = PromptTemplate.from_template(instruction_rag)
    retrieved_docs = ensemble_retriever.invoke(question)
    
    #print(len(retrieved_docs))
    # Filtrar documentos duplicados basados en el contenido
    #unique_docs = []
    #seen_contents = set()
    #for doc in retrieved_docs:
    #    content = doc.page_content  
    #    if content not in seen_contents:
    #        seen_contents.add(content)
    #        unique_docs.append(doc)

    context = format_docs(retrieved_docs)
    
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm_model_gemini
        | JsonOutputParser()
    )

    # Combine question and context
    combined_input = {
        "context": context,
        "question": question
    }

    try:
        result = rag_chain.invoke(combined_input)
    except Exception as e:
        # Si hay un error, retorna un mensaje por defecto.
        return {"respuesta": "Lo siento, hubo un problema al procesar la solicitud. Por favor, contacta con nosotros a través de nuestro WhatsApp dando clic aqui: [3223322322](https://wa.link/2gsvmz) o llamando al #322."}


    if isinstance(result, dict):
        parsed_result = result
    else:
        raise ValueError(f"Unexpected result type: {type(result)}")

    return parsed_result, context



#print(clasificador_rag("Quiero saber si con mi seguro hogar tengo servicio de plomeria de emergencia"))
print(clasificador_rag('¿Cómo puedo cambiar la dirección de mi póliza?,'))
#print(clasificador_rag("¿Como descargo una certificacion de mi seguro de cumplimiento?"))


"""

import pandas as pd
import time as time

# Cargar el archivo XLSX en un DataFrame
df = pd.read_excel('./data/evaluacion/AC_test.xlsx')

# Crear nuevas columnas vacías
df['output_ailab'] = ''
df['context_ailab'] = ''

# Aplicar la función a cada fila del DataFrame
for index, row in df.iterrows():
    time.sleep(0.5)
    
    mensaje_usuario = row['input']
    try:
        # Llamar a la función clasificador y asignar los resultados
        # Suponiendo que clasificador puede devolver una tupla de 2 o 3 elementos
        result_tuple = clasificador_rag(mensaje_usuario)

        # Desempaquetar el resultado y manejar si context está presente o no

        if len(result_tuple) == 2:
            result, context = result_tuple
            result = result["respuesta"]
            print(result)
        else:
            raise ValueError("La función 'clasificador' no devolvió el número esperado de valores.")

        # Almacenar los resultados en el DataFrame
        df.at[index, 'output_ailab'] = result
        df.at[index, 'context_ailab'] = context
    except ValueError as e:
        print(f"Error en la fila {index}: {e}")
        # Opcional: asignar valores por defecto o dejar en blanco
        df.at[index, 'output_ailab'] = None
        df.at[index, 'context_ailab'] = ''
    except Exception as e:
        # Manejar cualquier otro tipo de excepción
        print(f"Ocurrió un error inesperado en la fila {index}: {e}")
        df.at[index, 'output_ailab'] = None
        df.at[index, 'context_ailab'] = ''

# Guardar el DataFrame modificado en un nuevo archivo XLSX
output_file_path = './data/evaluacion/AC_test_ailab.xlsx'
df.to_excel(output_file_path, index=False)

print(f"El archivo Excel se guardó correctamente en {output_file_path}")
"""



