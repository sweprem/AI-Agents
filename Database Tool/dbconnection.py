from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.engine import URL
import logging

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.DEBUG)

connect_url = URL.create(
    'mysql+pymysql',
    username='root',
    password='',
    host='',
    port='3306',
    database='flightdata')

con=''
try:
    engine = create_engine(connect_url,connect_args={'connect_timeout': 60})
    print(engine)
    print("Engine created")
    with engine.connect() as con:
        con= engine.connect()
        print("connection created")
        rs = con.execute('SELECT * FROM flightdata.flights')
        print("Statement executed")
        for row in rs:
            print(row)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
finally:
    if 'con' in locals():
        con.close()
