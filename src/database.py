
import sys
sys.path.append('/opt/homebrew/lib/python3.11/site-packages')

import mysql.connector
from mysql.connector import Error
import pandas as pd

cursor = None

#function to create database
def create_database(connection, query):
    global cursor 
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as err:
        print(f"Error: '{err}'")

#function used to perform and implement queries on the database
def execute_query(connection, query):
    global cursor
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")

#function for reading data from database
def read_query(connection, query):
    global cursor
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as err:
        print(f"Error: '{err}'")

def add_user(connection, user_id, username):
    query = f"INSERT INTO Users (UserID, Username) VALUES ({user_id}, '{username}')"
    execute_query(connection, query)

#function to close database connection
def close_connection():
    global cursor
    cursor.close()
    database.close()

create_user_table = '''
    CREATE TABLE Users(
        UserID INT PRIMARY KEY,
        UserName VARCHAR(12)
    );
'''

create_EEGData_table = '''
    CREATE TABLE EEGData(
        UserID INT,
        EEGReading FLOAT,
        Timestamp TIMESTAMP,
        FOREIGN KEY (UserID) REFERENCES Users(UserID)
    );
'''

create_SleepStages_table = '''
    CREATE TABLE SleepStages(
        UserID INT,
        StageType VARCHAR(10),
        StartTimeStamp TIMESTAMP,
        EndTimeStamp TIMESTAMP,
        DurationInSeconds INT,
        FOREIGN KEY (UserID) REFERENCES Users(UserID)
    );
'''

create_Binaural_table = '''
    CREATE TABLE Binaural(
        UserID INT,
        LeftFrequency FLOAT,
        RightFrequency FLOAT,
        Timestamp TIMESTAMP,
        FOREIGN KEY (UserID) REFERENCES Users(UserID)
    )
'''

#establishes connection to SQL server
database = mysql.connector.connect(host = 'localhost', user = 'root', password = '')

#create new database
create_database(database, "CREATE DATABASE DreamCraftAI")

#execute querys to create database tables
execute_query(database, create_user_table)
execute_query(database, create_EEGData_table)
execute_query(database, create_SleepStages_table)
execute_query(database, create_Binaural_table)

#add users
add_user(database, 1, 'Trinity')
add_user(database, 2, 'Bradley')
