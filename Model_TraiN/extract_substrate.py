import pickle
def extractSubstrate(pickle_file):
    filehandler = open(pickle_file, 'rb')
    substrate = pickle.load(filehandler)
    return substrate