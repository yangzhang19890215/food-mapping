import pandas as pd
from pymongo import MongoClient

def connectDB():
    '''
    db = client.Food_Mapping
    :return: db
    '''
    client = MongoClient("vec-sim-005", 27017, maxPoolSize=50)
    db = client.Food_Mapping
    return db



def getCollection(collectionName):
    """
    df = getCollection(collectionName)
    :param collectionName: collectionName
    :return: df
    """
    db = connectDB()
    collection = db[collectionName]
    cursor = collection.find({})
    df = pd.DataFrame(list(cursor))
    if 'no_id' and '_id' in df:
        del df['_id']
    return df



def createCollection(mylist, collectionName):
    """
    createCollection(mylist = df.to_dict('records'), collectionName):
    mylist -> dictionary
    mylist = [{ "name": "Amy", "address": "Apple st 652"},
            { "name": "Hannah", "address": "Mountain 21"},
            { "name": "Michael", "address": "Valley 345"}]
        -> mylist = df.to_dict('records')
    check up more at https://www.w3schools.com/python/python_mongodb_getstarted.asp
    """
    db = connectDB()
    collection = db[collectionName]
    x = collection.insert_many(mylist)


def dropCollection(collectionName):
    db = connectDB()
    collection = db[collectionName]
    collection.drop()