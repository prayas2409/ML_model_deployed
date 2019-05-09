# Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/",methods=['GET','POST'])
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict(exp=None):
    if request.method == "POST":
        # Get the data from the POST request.

        # data = request.get_json(force=True)

        data = request.form['exp']
        # Make prediction using model loaded from disk as per the data.
        prediction = model.predict(np.array(float(data)).reshape(-1, 1))
        # Take the first value of prediction
        output = prediction[0]

        # if we want to use json Send in below way and then you'll have to parse it at html page using js or jquery
        # output=jsonify(output)
        return render_template('predict.html', salary=output)
    else:
        return str(model.predict(np.array(exp).reshape(-1, 1)))


if __name__ == '__main__':
    app.run(port=5000, debug=True)
