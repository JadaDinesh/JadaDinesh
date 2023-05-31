from utilities.db_connection import DB


class SQLLiteUtils:

    def __init__(self, string_connection):
        self.connection = DB.get_db(string_connection=string_connection)

    def put_df(self, df, table, string_connection):
        for i in range(0, 3):
            try:
                df.to_sql(name=table, con=self.connection, index=False, if_exists='replace')
                break
            except Exception as e:
                print("An exception has occurred as : {0}".format(e))
                self.connection = DB.reset_db_conn(string_connection=string_connection)
                continue
