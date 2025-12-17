from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

db_user = "gBN95a29"
db_password = "u[>xf*lKSO525!66M7GUQ"
db_host = "185.204.170.142"
db_port = "1433"
db_name = "master"

encoded_password = quote_plus(db_password)
connection_string = f"mssql+pyodbc://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes&timeout=30"
engine = create_engine(connection_string, connect_args={"timeout": 30})

with engine.connect() as conn:
    # List all databases
    print("=== Databases ===")
    result = conn.execute(text("SELECT name FROM sys.databases"))
    for row in result:
        print(f"  - {row[0]}")
    
    # List tables in current database
    print(f"\n=== Tables in '{db_name}' ===")
    result = conn.execute(text("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'"))
    for row in result:
        print(f"  - {row[0]}.{row[1]}")
