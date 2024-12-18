import pandas as pd
from chromadb import Client

def get_dataframe_from_chroma(client: Client, collection_name: str) -> pd.DataFrame:
    """
    Obtiene un DataFrame desde una colección de Chroma.

    Args:
        client (Client): Cliente de Chroma conectado a la base de datos.
        collection_name (str): Nombre de la colección en Chroma.

    Returns:
        pd.DataFrame: DataFrame con los datos de la colección.
    """
    # Obtener la colección donde están tus documentos
    collection = client.get_collection(collection_name)

    # Recuperar todos los documentos
    results = collection.get()

    # Extraer los campos necesarios
    ids = results['ids']
    documents = results['documents']
    metadatas = results['metadatas']

    # Asegurar que todos los IDs son cadenas de texto
    ids = [str(id_) for id_ in ids]

    # Crear listas para los datos del DataFrame
    data = []

    for i in range(len(ids)):
        # Extraer metadatos individuales
        meta = metadatas[i]

        # Crear el diccionario de datos para el DataFrame
        entry = {
            #'ids': ids[i],
            'Pregunta': meta.get("Pregunta", ""),
            'Respuesta': meta.get("Respuesta", "")
            #'Metadata': meta.get("Metadata", "")
        }

        data.append(entry)

    return pd.DataFrame(data)
