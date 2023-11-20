from flask import Flask,render_template,request

import pickle
import pandas as pd
import numpy as np

main_model = pickle.load(open('model.pkl','rb'))
scalar = pickle.load(open('scaler.pkl','rb'))

#Initialize our model
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict.html', methods=['GET'])
def predict():
   return render_template('predict.html')

@app.route("/predict",methods = ['POST'])
def predict1():
   
   brand = request.form['Brand']
   model = request.form['Model']
   os = request.form['OS']
   connectivity = request.form['Connectivity']
   displayType = request.form['DisplayType']
   resolution = request.form['Resolution']
   displaySize = request.form['DisplaySize']
   hrm = request.form['HRM']
   waterresistance = request.form['waterResistance']
   batterylife = request.form['batteryLife']
   gps = request.form['GPS']
   nfc = request.form['NFC']

   variables = [brand, model, os,connectivity,displayType,displaySize,resolution,waterresistance,batterylife,hrm,gps,nfc]

   x = [[float(var) for var in variables]]

   x_sclaed = scalar.transform(x)
   output = main_model.predict(x_sclaed)
   print(output)

   return render_template('watch_prediction.html', result="Based On Your Inputs The Prediction is: " + str(np.round(output[0])))

if __name__ == '__main__':
    app.run(port=5000, debug=True)