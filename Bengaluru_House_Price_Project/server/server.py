from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/get_location_names')
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })

    return response
    return "Hello World!"



if __name__ ==  "__main__":
    print("Start Python Flask Server For Banglore Home Price Prediction.")
    app.run()