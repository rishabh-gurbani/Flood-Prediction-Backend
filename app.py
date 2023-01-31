from flask import Flask, request, jsonify
import pickle
import numpy as np

pickle.load(open('model.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "hello world"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    mon = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT',
           'NOV', 'DEC']
    months = [request.form.get(x) for x in mon]

    input_query = np.array(months, dtype=float).reshape(1, 12)
    prediction = model.predict(input_query)[0]

    return jsonify({'prediction': str(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
