import pickle
import os

DB_FILE = "voices.pkl"

def load():
    return pickle.load(open(DB_FILE, 'rb')) if os.path.exists(DB_FILE) else {}

def save(db):
    pickle.dump(db, open(DB_FILE, 'wb'))

def add(name, embedding):
    db = load()
    db[name] = embedding
    save(db)

def get_all():
    return load()