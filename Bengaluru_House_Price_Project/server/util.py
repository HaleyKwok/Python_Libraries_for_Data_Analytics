import json 
import pickle 
import numpy as np

# global variables
__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower()) 
    except:
        loc_index = -1 # if location is not found, return -1
         
    x = np.zeros(len(__data_columns)) # create a numpy array of zeros of size len(__data_columns)
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return __model.predict([x])

'''

predict price model from Project.ipynb
def predict_price(location, sqft, bhk, bath):
    loc_index = np.where(X.columns == location)[0][0] 

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bhk
    x[2] = bath
    if loc_index >= 0:
        x[loc_index] = 1 
    return lr.predict([x])[0]

'''

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...\n")
    global __data_columns
    global __locations

    # open the file
    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:] # first 3 columns are "total_sqft", "bath", "bhk", and [3:] starts with "location" 

    with open("./artifacts/banglore_home_prices_model.pickle", "rb") as f:
        __model = pickle.load(f)
    print("loading saved artifacts...done\n")

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())