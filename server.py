# Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def index():
    # render_template("index.html")
    return "Hey world, please type predict/ experience value on the url"


@app.route('/predict/<float:exp>', methods=['GET','POST'])
def predict(exp):
    if request.method == "POST":
        # Get the data from the POST request.
        data = request.get_json(force=True)
        # Make prediction using model loaded from disk as per the data.
        prediction = model.predict([[np.array(data['exp'])]])
        # Take the first value of prediction
        output = prediction[0]
        return jsonify(output)
    else:
        return str(model.predict(np.array(exp).reshape(-1, 1)))


if __name__ == '__main__':
    app.run(port=5000, debug=True)
