from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

model_file = open('LinearRegressionPDRB.pkl', 'rb')
model_linreg = pickle.load(model_file, encoding='bytes')
model_lstm = tf.keras.models.load_model('model_lstm.h5')
model_multivariate = tf.keras.models.load_model('modelMultivariateLSTM.h5')
scaller_file = open('scaller.pkl', 'rb')
scaller = pickle.load(scaller_file, encoding='bytes')
scaller_file2 = open('scalerMultivariate.pkl', 'rb')
scallerMultivariate = pickle.load(scaller_file2, encoding='bytes')

@app.route('/')
def index():
    return render_template('index0.html')

@app.route('/linreg')
def linreg():
    return render_template('index.html', pdrb=0)

@app.route('/linreg', methods=['POST'])
def linregPredict():
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''
    tahun = int(request.form.get('Tahun'))
    
    prediction = model_linreg.predict([[tahun]])
    output = prediction[0]

    return render_template('index.html', pdrb=output, tahun=tahun)

@app.route('/lstm')
def lstm():
    return render_template('index1.html', pdrb=0)

@app.route('/lstm', methods=['POST'])
def lstmPredict():
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''
    data = request.form.getlist('Hari')
    data = list(map(int, data))
    data = np.reshape(data, (-1,1))
    data = scaller.transform(data)
    data = np.reshape(data, (1,30,1))
    
    prediction = model_lstm.predict(data)
    output = prediction
    output = scaller.inverse_transform(output)[0][0]

    return render_template('index1.html', pdrb=output)

@app.route('/multivariate')
def multivariate():
    return render_template('index2.html', pdrb=0)

@app.route('/multivariate', methods=['POST'])
def multivariatePredict():
    '''
    Predict the GDP cost based on user inputs
    and render the result to the html page
    '''
    data = request.form.getlist('Sektor')
    data = list(map(int, data))
    data = np.reshape(data, (-1,1))
    data = np.reshape(data, (1,8))
    data = scallerMultivariate.transform(data)
    data = np.reshape(data, (1,1,8))

    prediction = model_multivariate.predict(data)
    data = np.reshape(data, (1,8))
    prediction = np.concatenate((data[:, :-1],prediction), axis=1)
    prediction = scallerMultivariate.inverse_transform(prediction)
    prediction = prediction[:,-1]
    output = prediction[0]
    return render_template('index2.html', pdrb=output)

if __name__ == '__main__':
    app.run(debug=True)