from flask import Flask, request, jsonify
from logging import FileHandler, WARNING
import pickle
import numpy as np

pickle.load(open('model.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)
file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)


@app.route('/')
def home():
    return "hello world"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    mon = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT',
           'NOV', 'DEC']
    months = [request.form.get(x) for x in mon]

    input_query = np.array(months, dtype=int).reshape(1, 12)
    prediction = model.predict(input_query)[0]

    return jsonify({'flood': str(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
