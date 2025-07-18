import pandas as pd
from flask import Flask
import pickle
import json
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    #load model
    filename = 'model/prediction_model.pkl'
    model = pickle.load(open(filename, 'rb'))

    with open('inputs/inputs/inputs.json', 'r') as f:
        user_input = json.load(f)

    rooms = user_input['rooms']
    sqft = user_input['sqft']

    user_input_predict = np.array([[rooms,sqft]])
    predicted_model = model.predict(user_input_predict)

    #output
    output = {"The predicted output is": float(predicted_model[0])}

    with open('outputs/outputs/outputs.json', 'w') as f:
        json.dump(output,f)
        
    print("The predicted output is", output)
    return output


if __name__ == '__main__':
    app.run(port=5000, debug=True)