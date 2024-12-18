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

from utils.paths import embedding_model, client, collection_name_ac, collection_name_sb, llm_model_gemini, llm_model_gemini_pro, llm_model_gpt_4, llm_model_gpt_35
from utils.get_vectorstore import get_dataframe_from_chroma

collection_name = collection_name_sb

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
    return "\n\n".join(f"""Pregunta: {doc.metadata['Pregunta']}
                    Respuesta: {doc.metadata['Respuesta']}"""
                    for doc in docs)


instruction_rag = """
Eres Olivia, un agente virtual experto en evaluar y clasificar interacciones entre clientes y agentes para Seguros Bolívar. Tu tarea principal es analizar la interacción según las reglas y criterios estrictos descritos a continuación, proporcionando siempre una justificación clara y fundamentada para tu calificación.
## Reglas de Evaluación:
1. **Elementos a analizar**:
   - **Pregunta del cliente**: Consulta inicial realizada por el cliente.
   - **Contexto proporcionado**: Una lista detallada de preguntas frecuentes y sus respuestas correctas, que sirven como referencia oficial para evaluar.
   - **Respuesta del agente**: Respuesta emitida por el agente, que será comparada con el contexto proporcionado y la pregunta del cliente.

2. **Categorías de calificación**:
    **"bien"**:  
        - La **pregunta del cliente** coincide con alguna pregunta en el contexto.
        - La **respuesta del agente** coincide con la información proporcionada en el contexto.
        - La respuesta del agente **NO** utiliza la respuesta predefinida.

    **"mal"**:  
        - La **pregunta del cliente** coincide con alguna pregunta en el contexto o hay información relevante en el contexto para responder.
        - La **respuesta del agente** **NO** coincide con la información proporcionada, aun cuando el contexto ofrecía suficiente información para responder correctamente.
        - La **respuesta del agente** **NO** utiliza la respuesta predefinida.
        
    **"Sin Coincidencia"**:  
        - La **pregunta del cliente** NO coincide con ninguna pregunta en el contexto, y no hay información relevante en el contexto para responder.  
        - La respuesta del agente es la respuesta predefinida, expresada como se detalla a continuación:
            > _"¡Lo siento! Aún estoy aprendiendo y no cuento con la información que necesita. Por favor, escríbanos a nuestro WhatsApp dando clic aquí: 3223322322 o llámenos al #322 para resolver sus dudas."_  
        
    **"Sin Coincidencia - mal"**:  
        - La **pregunta del cliente** **NO** coincide con ninguna pregunta en el contexto, y no hay información relevante en el contexto para responder la pregunta.
        - La **respuesta del agente** **NO** utiliza la respuesta predefinida.

3. **Formato de salida**:
   Devuelve el resultado en formato JSON con la siguiente estructura:  
   
   {{
     "calificacion": "[Una de las categorías: 'bien', 'mal', 'Sin Coincidencia', 'Sin Coincidencia - mal']"
     "justificacion": "[Proporciona una explicación concisa pero detallada de por qué se asignó esta calificación, incluyendo las coincidencias o discrepancias entre la pregunta, el contexto y la respuesta del agente.]"
   }}
   
Pregunta cliente: {question}
Contexto proporcionado: {context}
Respuesta del agente: {answer}
"""

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

# Cargar el dataset
df = get_dataframe_from_chroma(client, collection_name)

# Crear documentos utilizando SimpleDocument
docs = [SimpleDocument(row['Pregunta'], row['Respuesta']) for _, row in df.iterrows()]

# Crear el BM25 retriever
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4



# RAG CHAIN SE PUEDE USAR GEMINI (llm_model_gemini) 
def calificador(question, answer):
    chroma_retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k":4}
    )

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.3, 0.7])
    
    custom_rag_prompt = PromptTemplate.from_template(instruction_rag)
    retrieved_docs = ensemble_retriever.invoke(question)
    
    #print(len(retrieved_docs))
    # Filtrar documentos duplicados basados en el contenido
    unique_docs = []
    seen_contents = set()
    for doc in retrieved_docs:
        content = doc.page_content  # Asumiendo que `page_content` es el atributo único
        if content not in seen_contents:
            seen_contents.add(content)
            unique_docs.append(doc)

    context = format_docs(unique_docs)
    print(context)
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough(), "answer": RunnablePassthrough()}
        | custom_rag_prompt
        | llm_model_gemini_pro
        | JsonOutputParser()
    )

    # Combine question and context
    combined_input = {
        "context": context,
        "question": question,
        "answer": answer
    }

    try:
        result = rag_chain.invoke(combined_input)
    except Exception as e:
        # Si hay un error, retorna un mensaje por defecto.
        return {"respuesta": f"error, {e}"}


    #print(result)
    if isinstance(result, dict):
        parsed_result = result
    else:
        raise ValueError(f"Unexpected result type: {type(result)}")

    return parsed_result, context


q = "estoy tratando de comprar el soat"
a = "¡Lo siento! Aun estoy aprendiendo y no cuento con la información que necesita.\n\nPor favor, escribanos a nuestro WhatsApp dando clic aquí: [3223322322](https://wa.link/2gsvmz) o llamenos al #322 para resolver sus dudas. \n"
print(calificador(q,a))




