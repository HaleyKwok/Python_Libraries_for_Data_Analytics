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

    return round(__model.predict([x])[0],2) # return only one element from the saved artifacts with 2 decimal numbers

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



def load_saved_artifacts():
    print("loading saved artifacts...\n")
    global __data_columns
    global __locations

    # open the file
    with open("/Users/haleyk/Documents/Python_Libraries_for_Data_Analytics/Python_Libraries_for_Data_Analytics/Bengaluru_House_Price_Project/server/artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:] # first 3 columns are "total_sqft", "bath", "bhk", and [3:] starts with "location" 

    global __model
    if __model is None:
        with open("/Users/haleyk/Documents/Python_Libraries_for_Data_Analytics/Python_Libraries_for_Data_Analytics/Bengaluru_House_Price_Project/server/artifacts/banglore_home_prices_model.pickle", "rb") as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done\n")
    # with open("/Users/haleyk/Documents/Python_Libraries_for_Data_Analytics/Python_Libraries_for_Data_Analytics/Bengaluru_House_Price_Project/server/artifacts/banglore_home_prices_model.pickle", "rb") as f:
    #     __model = pickle.load(f) -> need global variable __model to be able to use it in the predict_price function
    # print("loading saved artifacts...done\n")

def get_data_columns():
    return __data_columns

def get_location_names():
    return __locations


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 2))
    print(get_estimated_price('Kalhalli', 1000, 3, 2)) # other location
    print(get_estimated_price('Ejipura', 1000, 3, 2)) # other location



     