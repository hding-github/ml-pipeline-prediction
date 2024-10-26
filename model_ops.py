import pickle

def save(model_xgb, file_name):
    # save
    pickle.dump(model_xgb, open(file_name, "wb"))
    return True

def load(file_name):
    # load
    model_xgb = pickle.load(open(file_name, "rb"))
    return model_xgb