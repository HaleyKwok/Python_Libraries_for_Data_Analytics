from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/hello')
def hello():
    return "Hello World!"

if __name__ ==  "__main__":
    print("Start Python Flask Server For Banglore Home Price Prediction.")
    app.run()