import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
from typing import List
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.tools import tool
from operator import itemgetter

load_dotenv()

class Table(BaseModel):
    """
    Represents a table in the SQL database.

    Attributes:
        name (str): The name of the table in the SQL database.
    """
    name: str = Field(description="Name of table in SQL database.")


class HospitalSQLAgent:
    """
    A specialized SQL agent that interacts with the Hospital SQL Server database using an LLM.
    
    The agent handles SQL queries by first identifying relevant tables from the hospital database
    (Patients, Doctor, Appointments, Diagnosis, PatientDiagnosis, ClinicalNotes) and then
    generating SQL Server compatible queries.
    
    Attributes:
        sql_agent_llm (ChatOpenAI): The language model used for SQL generation.
        table_extractor_llm (ChatOpenAI): The language model used for table extraction.
        db (SQLDatabase): The SQL Server database connection.
        full_chain (Runnable): Complete chain for table extraction and query generation.
    """

    def __init__(self, sqldb_uri: str, llm: str, llm_temperature: float) -> None:
        """Initialize the Hospital SQL Agent with SQL Server connection."""
        
        # Initialize LLMs
        self.sql_agent_llm = ChatOpenAI(
            model=f"openai/{llm}", 
            temperature=llm_temperature,
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("OPEN_AI_API_KEY")
        )
        self.table_extractor_llm = ChatOpenAI(
            model="openai/gpt-4o-mini", 
            temperature=0,
            base_url="https://models.github.ai/inference",
            api_key=os.getenv("OPEN_AI_API_KEY")
        )
        
        # Connect to SQL Server database
        self.db = SQLDatabase.from_uri(sqldb_uri)
        table_names = "\n".join(self.db.get_usable_table_names())
        
        # Create table extraction chain
        system = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_names}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""
        
        table_chain = create_extraction_chain_pydantic(
            pydantic_schemas=Table, 
            llm=self.table_extractor_llm, 
            system_message=system
        )
        
        # Create SQL Server query generation chain
        sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL Server expert. Generate ONLY valid SQL Server queries.

CRITICAL RULES:
- NEVER use backticks (`) - they cause syntax errors in SQL Server
- Use square brackets [column_name] for identifiers with spaces/special chars
- For simple identifiers, no quotes needed
- Use TOP N instead of LIMIT N
- Use SQL Server functions and syntax
- Return only the SQL query, no markdown, no explanation

Database dialect: SQL Server
Available tables: {table_info}"""),
            ("human", "Question: {input}\n\nBased on the table schema above, write a SQL Server query to answer this question.\nUse only the following tables: {table_names_to_use}")
        ])
        
        def create_custom_sql_server_chain(llm, db):
            def get_table_info():
                return db.get_table_info()
            
            return (
                {
                    "input": lambda x: x["question"],
                    "table_info": lambda x: get_table_info(),
                    "table_names_to_use": lambda x: x.get("table_names_to_use", [])
                }
                | sql_prompt 
                | llm 
                | StrOutputParser()
            )
        
        def extract_table_names(tables):
            """Extract table name strings from Table objects"""
            if isinstance(tables, list):
                return [table.name if hasattr(table, 'name') else str(table) for table in tables]
            return []
        
        query_chain = create_custom_sql_server_chain(self.sql_agent_llm, self.db)
        
        # Create complete chain
        table_chain_with_extraction = (
            {"input": itemgetter("question")} 
            | table_chain 
            | extract_table_names
        )
        
        self.full_chain = RunnablePassthrough.assign(
            table_names_to_use=table_chain_with_extraction
        ) | query_chain


@tool
def query_hospital_database(query: str) -> str:
    """
    Query the Hospital SQL Server Database to extract information about patients, doctors, appointments, diagnoses, and clinical notes.
    
    Use this tool to answer questions about:
    - Patient information and demographics
    - Doctor profiles and specializations
    - Appointment scheduling and history
    - Medical diagnoses and patient diagnosis records
    - Clinical notes and observations
    - Hospital statistics and reports
    
    Input should be a natural language question about hospital data.
    """
    from agent_graph.load_tools_config import LoadToolsConfig
    
    TOOLS_CFG = LoadToolsConfig()
    
    # Build SQL Server URI
    server = os.getenv("SQL_SERVER")
    database = os.getenv("SQL_DATABASE") 
    username = os.getenv("SQL_USERNAME")
    password = os.getenv("SQL_PASSWORD")
    driver = os.getenv("SQL_ODBC_DRIVER", "ODBC Driver 17 for SQL Server")
    
    odbc_str = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=Yes;Encrypt=No;"
    db_uri = f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_str)}"
    
    # Create agent and execute query
    agent = HospitalSQLAgent(
        sqldb_uri=db_uri,
        llm=TOOLS_CFG.hospital_sqlagent_llm,
        llm_temperature=TOOLS_CFG.hospital_sqlagent_llm_temperature
    )
    
    sql_query = agent.full_chain.invoke({"question": query})
    return agent.db.run(sql_query)