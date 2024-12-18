import os
import pandas as pd
from langchain_community.vectorstores import Chroma
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__)) 
root_dir = os.path.dirname(os.path.dirname(current_dir)) 
sys.path.append(root_dir)


from utils.paths import path_data_ac, path_data_sb, embedding_model, client, collection_name_ac, collection_name_sb

# Cargar el DataFrame
df = pd.read_csv(path_data_sb) #se debe cambir porque hay que poner uno a uno sb y ac
collection_name = collection_name_ac

# Verificar si 'ids' no existe y crearla si es necesario
if 'ids' not in df.columns:
    df['ids'] = [f"id_{i}" for i in range(1, len(df) + 1)]  # Crear IDs únicos

df['ids'] = df['ids'].astype(str)

# Procesar los textos y generar las listas con los embeddings
docs, metas = [], []  # listas docs (item), datos


for index, row in df.iterrows():
    
    meta_dict = {
        "Pregunta": row["Pregunta"],
        "Respuesta": row["Respuesta"],
        "Metadata": row["Metadata"]
    }
    doc = row['Pregunta']

    # Guardar resultados
    metas.append(meta_dict)
    docs.append(doc)
    

ids = df['ids'].tolist()

#print(len(ids))
#print(len(metas))
#print(len(docs))


#Crear embeddings (No hay errores de requests)
embs = embedding_model.embed_documents(docs)



# Crear colección con la función de distancia de coseno
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)

# Agregar las observaciones a la Vector store
collection.add(
    documents=docs,
    embeddings=embs,
    metadatas=metas,
    ids=ids
)



# Prueba de un query a la base ingestada 
vector_store = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=embedding_model,
)


print("Ahora hay", vector_store._collection.count(), "documentos en la coleccion")
