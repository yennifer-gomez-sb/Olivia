import os
import chromadb
from langchain_openai import AzureOpenAIEmbeddings
from google.cloud import secretmanager
from langchain_openai import AzureChatOpenAI
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import ChatVertexAI

sm_client = secretmanager.SecretManagerServiceClient()
vs_secrets = 'projects/709427406268/secrets/vector_store_creds/versions/latest'
response = sm_client.access_secret_version(name=vs_secrets)
host = response.payload.data.decode("UTF-8")

openai_secrets = 'projects/709427406268/secrets/Azure_Keys_OpenAI_prod/versions/latest'
response = sm_client.access_secret_version(name=openai_secrets).payload.data.decode("UTF-8")
keys = response.splitlines()

for key in keys:
    name, value = key.split('=')
    os.environ[name] = value



# Parametros
#collection_name_ac = 'base_conocimientos_olivia_ac_2' 
#collection_name_sb = 'base_conocimientos_olivia_sb_2' 
collection_name_ac = 'base_conocimientos_olivia_ac_open' 
collection_name_sb = 'base_conocimientos_olivia_sb_open' 
tenant = 'dev'
database = 'topicos-ailab'
ip = f"http://{host}:8000"



# Modelo de Embeddings
embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("EMBEDDINGS_NAME"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
)


# Cliente de Chroma
client = chromadb.HttpClient(
    host= ip,
    tenant= tenant,
    database= database
)


# Lectura de la ultima version de data
#path_data_ac = '/home/yennifer-gomez/ailab-olivia-faqs/data/expanded/data_ac_I.csv'
#path_data_sb = '/home/yennifer-gomez/ailab-olivia-faqs/data/expanded/data_sb_I.csv'
path_data_ac = '/home/yennifer-gomez/ailab-olivia-faqs/data/expanded/data_ac_open.csv'
path_data_sb = '/home/yennifer-gomez/ailab-olivia-faqs/data/expanded/data_sb_open.csv'

# LLM GPT
llm_model_gpt_35 = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment=os.getenv("GPT35_NAME"),
)

llm_model_gpt_4 = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_deployment=os.getenv("GPT4_NAME"),
)

# LLM GEMINI
llm_model_gemini = VertexAI(model_name = 'gemini-1.5-flash-002',  temperature = 0.2, top_k=40,top_p=0.95)
llm_model_gemini_pro = VertexAI(model_name = 'gemini-1.5-pro',  temperature = 0.0)
llm_model_gemini_flash = VertexAI(model_name = 'gemini-2.0-flash-exp',  temperature = 0.2, top_k=40,top_p=0.95)


