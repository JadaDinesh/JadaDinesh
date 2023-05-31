from operator import delitem
import os
from os import path
import random
import sqlite3 as sql
import timeit

import numpy as npy
import pandas as pnd

def data_load():
    result = []
    with open("ideal.csv") as f:
        for line in f.readlines():
            result.append(line.split(","))
    return result

def adding_record(lst):
    row = random.sample(range(-30, 30), 51)
    lst.append(row)

def getting_record(result):
    x = random.randint(-200, 199) / 10
    for row in result:
        if row[0] == str(x):
            return row


#NumPy
def data_load():
    return npy.loadtxt("ideal.csv", delimiter=",", skiprows=1)

def adding_record(lst):
    row = random.sample(range(-30, 30), 51)
    return npy.append(lst, row)

def getting_record(data):
    x = random.randint(-200, 199) / 10
    result = npy.where(data[:,0] == x)
    if len(result[0]) > 0:
        index = result[0][0]
        return data[index]


#pandas

def data_load():
    return pnd.read_csv("ideal.csv", sep=",")

def adding_record(data):
    row = random.sample(range(-30, 30), 51)
    data.loc[data.shape[0]] = row

def getting_record(data):
    x = random.randint(-200, 199) / 10
    result = data.loc[data["x"] == x]
    if len(result) > 0:
        return result

#Sqlite
'''
def data_load_no_delete():
    conn = sql.connect("database.db")
    data = pnd.read_csv("ideal.csv")
    data.to_sql("ideal", con=conn, index=False)

def create_db():
    conn = sql.connect("database.db")
    os.remove("database.db")

def data_load():
    conn = sql.connect("database.db")
    data = pnd.read_csv("ideal.csv")
    data.to_sql("ideal", con=conn, index=False)
    os.remove("database.db")

def add_record():
    row = tuple(random.sample(range(-30, 30), 51))
    con = sql.connect("database.db")
    cur = con.cursor()
    qm = ["?" for i in range(51)]
    cur.execute("insert into ideal values ({})".format(", ".join(qm)), row)

def get_record():
    x = random.randint(-200, 199) / 10
    con = sql.connect("database.db")
    cur = con.cursor()
    cur.execute("select * from ideal where x=:x", {"x": x})'''


if __name__=="__main__":
    print(timeit.timeit(
        "data_load()",
        setup="from __main__ import data_load",
        number=1000))
    print(timeit.timeit(
        "adding_record(result)",
        setup="from __main__ import data_load, add_record;result = data_load()",
        number=1000))
    print(timeit.timeit(
        "getting_record(result)",
        setup="from __main__ import data_load, get_record;result = data_load()",
        number=1000))
