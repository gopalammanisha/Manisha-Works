from flask import Flask, render_template, request
import pickle
import numpy as np
import xgboost


import pandas as pd


filename = 'xgmodel.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        gravity = float(request.form['gravity'])
        ph = float(request.form['ph'])
        osmo = int(request.form['osmo'])
        cond = int(request.form['cond'])
        urea = int(request.form['urea'])
        calc = float(request.form['calc'])
        
        
        data = np.array([[gravity,ph,osmo,cond,urea,calc]])
        my_prediction = model.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
        
        

if __name__ == '__main__':
	app.run(debug=True)

