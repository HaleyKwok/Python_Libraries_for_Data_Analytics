import json 
import pickle 

# global variables
__locations = None
__data_columns = None
__model = None

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