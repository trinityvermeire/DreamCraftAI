
#import classes for flask connection
from flask import Flask, jsonify, request
import serial
import time
import mysql.connector

#rate of information transfer in a communication channel
BAUD_RATE = 115200
SERIAL_PORT = '-----'

#create an instance of the flask class
app = Flask(__name__)

#route is used to tell flask what URL should trigger the function
@app.route('/get_data', methods = ['GET', 'POST'])
#methods specifies which HTTP methods are allowed

#test function
def get_data():
    message = {'message': 'Hello from Python backend!'}
    return jsonify(message) 

#route to page where bineaural beats are produced
@app.route('/read_qtPY', methods = ['GET', 'POST'])'
#function that reads information from qtPY
def read_qtPY(UserID):
    try:
        #create instance of QTPY serial connection
        qtPY = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout = 0.1)
        print("Serial connection established")

        #establishes connection to SQL server
        database = mysql.connector.connect(host = 'localhost', user = 'root', password = 'vermeire')
        cursor = database.cursor()
        print("SQL connection established")

        #read data from qyPY
        while True:
            data = qtPY.readline().decode().strip()

            #checks if data is empty and if not prints data and timestamp
            if data:
                print(time.strftime('%H:%M:%S'), data)
                #add value to sql database
                sql = "INSERT INTO EEGData (UserID, EEGReading, Timestamp) VALUES (%s, %s, %s)
                time = int(time.time())
                values = (UserID, data, time)
                cursor.execute(sql, values)
                database.commit()
            else:
                print('no data!')

    #prints any error that occurs
    except serial.SerialException as e:
        print("Error", e)

    cursor.close()
    database.close()
    qtPY.close()
    
    

#ensures app is only ran when executed in terminal
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
    read_qtPY()
