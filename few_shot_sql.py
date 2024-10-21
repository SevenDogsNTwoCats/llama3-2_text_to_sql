from langchain_community.llms import Ollama
llm = Ollama(model = "llama3.2:1b")
llm.invoke("Hi")

# from langchain_community.utilities import SQLDatabase
# db = SQLDatabase.from_uri("sqlite:///chinook.db", sample_rows_in_table_info = 3)

from langchain_community.utilities import SQLDatabase
import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Parámetros de conexión a MySQL
user = os.getenv("USER")
password = os.getenv("PASSWORD")
host = os.getenv("HOST")  # O el host donde esté tu base de datos
port = os.getenv("PORT")  # El puerto estándar de MySQL
database = os.getenv("DATABASE")

# Crear la URI de conexión para MySQL
uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"

# Conectar a la base de datos MySQL
db = SQLDatabase.from_uri(uri, sample_rows_in_table_info=3)

print(db.table_info)
prompt = [
  "input: ¿Cuáles son los códigos de alarma que más se repiten en los medidores?",
  "output: SELECT ALARM_CODE, COUNT(*) AS frecuencia FROM pnrp.airflow_hexing_alarmas GROUP BY ALARM_CODE ORDER BY frecuencia DESC;",
  "input: ¿Qué medidores han generado alarmas en los últimos 7 días?",
  "output: SELECT MEDIDOR, ALARM_CODE, ALARM_DESC, FECHA FROM pnrp."
]

# Inicializar el array de JSON
queries = []

# Iterar sobre los elementos del array de dos en dos (pares de input y output)
for i in range(0, len(prompt), 2):
    input_str = prompt[i].replace("input: ", "").strip()  # Limpiar el texto 'input:'
    output_str = prompt[i + 1].replace("output: ", "").strip()  # Limpiar el texto 'output:'
    
    # Crear el objeto JSON
    query = {
        "input": input_str,
        "query": output_str
    }
    
    # Añadir al array
    queries.append(query)

# Mostrar el resultado
import json
print(json.dumps(queries, indent=2))


examples = queries
print(len(examples))
# primer ejemplo
print(examples[0])

from langchain_community.embeddings import OllamaEmbeddings

embeddings = (
    OllamaEmbeddings(model = "llama3.2:1b")
)

from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    FAISS,
    k=3,
    input_keys=["input"],
    )

matched_queries = example_selector.vectorstore.search("¿Qué alarmas tiene el medidor 'MEDIDOR789' durante su periodo de análisis para el circuito 'BVI211'?", search_type = "mmr")
print(matched_queries)

for doc in matched_queries:
    print(doc.page_content)

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool, InfoSQLDatabaseTool, ListSQLDatabaseTool, QuerySQLCheckerTool

sql_db_query =  QuerySQLDataBaseTool(db = db)
sql_db_schema =  InfoSQLDatabaseTool(db = db)
sql_db_list_tables =  ListSQLDatabaseTool(db = db)
sql_db_query_checker = QuerySQLCheckerTool(db = db, llm = llm)

tools = [sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker]

for tool in tools:
    print(tool.name + " - " + tool.description.strip() + "\n")

system_prefix = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Here are some examples of user inputs and their corresponding SQL queries:

""" 

suffix = """
Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate

dynamic_few_shot_prompt_template = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input"],
    prefix=system_prefix,
    suffix=suffix
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=dynamic_few_shot_prompt_template),
    ]
)

prompt_val = full_prompt.invoke(
    {
        "input": "¿Qué alarmas tiene el medidor 'MEDIDOR789' durante su periodo de análisis para el circuito 'BVI213'?",
        "tool_names" : [tool.name for tool in tools],
        "tools" : [tool.name + " - " + tool.description.strip() for tool in tools],
        "agent_scratchpad": [],
    }
)
print(prompt_val.to_string())

output_string = prompt_val.to_string()

# Extraer la consulta SQL utilizando una búsqueda de texto
# Busca la línea que comienza con "SQL query:" y extrae el texto siguiente.
sql_query = ""
lines = output_string.split('\n')
for line in lines:
    if line.startswith("SQL query:"):
        sql_query = line.split("SQL query:")[1].strip()  # Obtener la consulta SQL
        break

# Imprimir solo la consulta SQL
print(sql_query)
