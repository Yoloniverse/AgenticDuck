from langchain.agents import tool
from langchain_tavily import TavilySearch
from typing import List
import logging
import os
from dotenv import load_dotenv

# get_staff_info
from sqlalchemy import create_engine, text, inspect
import json
#print("CWD:", os.getcwd())
######################################################################
#                             Save Log                               #
######################################################################
#load_dotenv()

# main에서 만든 것과 동일한 이름 사용
logger = logging.getLogger("Agent")

# db_url = os.environ.get('DB_URL')
# db_name = "langgraph"
# print(f'{db_url}{db_name}')

@tool
def get_doc_apprl_info(query: str) -> str:
    """Search for instructions and procedures on using the internal electronic approval system (Amaranth)"""

    answer = ""
    return answer


@tool
def get_staff_info(query: str) -> str:
    """Search for the company’s organizational chart and employee contact directory"""
    #sampleDB
    db_user = "admin"
    db_password = "sdt251327"
    db_host = "127.0.0.1"
    db_name = "langgraph" 

    # Const`ruct the connection string
    connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    engine = create_engine(connection_string)

    # Test the connection using raw SQL
    # with engine.connect() as connection:
    #     result = connection.execute(text("select * from STAFF_INFO limit 3"))
    #     for row in result:
    #         print(row)

    connection = engine.connect()
    inspector = inspect(engine)
    prompt_db_structure= """"""
    # 스키마 순회
    for schema_name in inspector.get_schema_names():
        if schema_name == db_name:
            for table_name in inspector.get_table_names(schema=schema_name):
                # 컬럼 이름과 타입 수집
                columns = inspector.get_columns(table_name, schema=schema_name)
                column_list = [
                    f"{col['name']}"
                    for col in columns
                ]
                prompt_db_structure += f'1.table_name: {table_name}\n2.columns: {column_list}'

    #print(prompt_db_structure)
    #db_structure_str = json.dumps(db_structure,  ensure_ascii=False)


    answer = ""
    return answer

