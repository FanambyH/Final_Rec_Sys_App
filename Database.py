import sqlite3
import pandas as pd

from sqlite3 import Error

class Database():
    """
    This class contains functions necessary to interact with the database 
    """

    def __init__(self):
        self.con = sqlite3.connect(r"recommender_system.sqlite3")
        self.cur = self.con.cursor()

    def query_table(self):
        """
        This function queries all the table names.

                Parameters:
                        -

                Returns:
                        -
        """
        self.cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        print(self.cur.fetchall())

     ############## CREATE QUERY ###################
    def create_feedback_table(self):
        """
        This function allows to create the feedback table.

                Parameters:
                        -

                Returns:
                        -
        """
        sentiment_sql = """
        CREATE TABLE feedback (
            id_fb integer PRIMARY KEY AUTOINCREMENT,
            title text NOT NULL,
            feedback text NOT NULL)"""
        self.cur.execute(sentiment_sql)
    
     ############## INSERT QUERY  ###################

    def insert_new_feedback_query(self,title,feedback):
        """
        This function allows to insert new feedback data into the feedback table .

                Parameters:
                        -

                Returns:
                        the last row of the table
        """
        sql = 'INSERT INTO feedback (title,feedback) VALUES(?,?)'
        self.cur.execute(sql, (title,feedback))
        return self.cur.lastrowid

    ############## READ QUERY  ###################

    def get_feedback_list(self):
        """
        This function allows to access the list of feedback.

                Parameters:
                        -

                Returns:
                        df(DataFrame):the list of all feedbacks
        """
        # Store the result of SQL query  in the dataframe
        df = pd.read_sql_query("SELECT * FROM feedback", self.con)
        
        return df

    ################### POPULATE TABLE ###################
    def populate_feedback_table(self,df):
        """  This function allows to allows to populate the feedback table.

                Parameters:
                        -

                Returns:
                        -
        """
        try:
            for i in range(df.shape[0]):
                #insert into the table 
                fb = self.insert_new_feedback_query(df['title'].iloc[i],df['feedback'].iloc[i])
                # commit the statements
                self.con.commit()
        except:
            # rollback all database actions since last commit
            self.con.rollback()
            raise RuntimeError("An error occurred ...")


def main():
    """  This is the main function that allows to test 
         the database functionality apart from the system.

                Parameters:
                        -

                Returns:
                        -
    """
    db = Database() 
    db.create_feedback_table()
    db.query_table()
    df_fb = pd.DataFrame(columns=['title','feedback'])
    df_fb['title'] =  ['test_1.wav','test_2.wav']
    df_fb['feedback'] = ['low','high']
    print(df_fb)

    # populate feedback table 
    db.populate_feedback_table(df_fb)
        
    df_feedback = db.get_feedback_list()
    print(df_feedback)


if __name__ == "__main__":
    main()
