import pymysql
import traceback
try:
    print("Attempting direct MySQL connection...")
    conn = pymysql.connect(
        host="",
        user="root",
        password="",
        port=3306  )
    print("✅ Direct connection successful!")
    conn.close()
except pymysql.MySQLError as err:
    print(f"❌ MySQL Connector Error: {err}")
    traceback.print_exc()
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
    traceback.print_exc()
