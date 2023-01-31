from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    return "hello world"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    mon = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT',
           'NOV', 'DEC']
    months = [request.form.get(x) for x in mon]
    modelType = [request.form.get('MODEL')]

    input_query = np.array(months, dtype=float).reshape(1, 12)

    pickle.load(open('LR.pkl', 'rb'))
    model = pickle.load(open('LR.pkl', 'rb'))

    if modelType == 'LR':
        pickle.load(open('LR.pkl', 'rb'))
        model = pickle.load(open('LR.pkl', 'rb'))
    elif modelType == 'RF':
        pickle.load(open('RF.pkl', 'rb'))
        model = pickle.load(open('RF.pkl', 'rb'))
    elif modelType == 'SVM':
        pickle.load(open('SVM.pkl', 'rb'))
        model = pickle.load(open('SVM.pkl', 'rb'))

    prediction = model.predict(input_query)[0]

    print(modelType)

    return jsonify({'prediction': str(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
