import pyodbc

# Connection details
db_user = "gBN95a29"
db_password = "u[>xf*lKSO525!66M7GUQ"
db_host = "185.204.170.142"
db_port = "1433"
db_name = "master"

# Try direct pyodbc connection with explicit connection string
conn_str = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={db_host},{db_port};"
    f"DATABASE={db_name};"
    f"UID={db_user};"
    f"PWD={db_password};"
    f"TrustServerCertificate=yes;"
    f"Connection Timeout=30;"
)

print("Attempting connection...")
print(f"Server: {db_host}:{db_port}")
print(f"Database: {db_name}")
print(f"User: {db_user}")
print("-" * 40)

try:
    conn = pyodbc.connect(conn_str, timeout=30)
    print("SUCCESS! Connected to database.")
    
    cursor = conn.cursor()
    cursor.execute("SELECT @@VERSION")
    row = cursor.fetchone()
    print(f"SQL Server Version: {row[0][:50]}...")
    
    conn.close()
except pyodbc.Error as e:
    print(f"Connection failed: {e}")
