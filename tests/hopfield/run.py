import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from toolz.functoolz import pipe

set_name = 'large-25x25.csv'
data_path = 'data/hopfield/patterns/' +set_name
df = pd.read_csv(data_path)
size = pipe( re.search('-(\d+)x(\d+)',set_name).groups(), (lambda t: (int(t[1]),int(t[0]))))
imgs = [(ind,np.resize(series,size)) for ind, series in df.iterrows()]


import hopfield.hebbian
from importlib import reload  
hebbian = reload(hopfield.hebbian)

def get_predict(synch, dataframe,df_size):
    elements = df_size[0]*df_size[1]
    hebb = hebbian.hebbian(elements,synch)
    hebb.train([series for ind, series in dataframe.iterrows()])
    return hebb.predict

hebb_sync = get_predict(True,df,size)
hebb_async = get_predict(False,df,size)

id = (lambda i: i)
func = [id, hebb_sync, hebb_async]


def predict_and_save_img(series,predict, offset,i):
    predicted = predict(series)
    item = (len(df)*offset+i,np.resize(predicted,size))
    return item

imgs=[]
org = {fi:[] for fi,f in enumerate(func)}
for fi,f in enumerate(func):
    for i ,series in df.iterrows():
        imgs.append(predict_and_save_img(series,f,fi,i))

mod = {fi:[] for fi,f in enumerate(func)}
for fi,f in enumerate(func):
    for i ,series in df.iterrows():
        half = int((size[0]*size[1])/2)
        series = pd.concat([0*series[0:half],series[half:]])
        imgs.append(predict_and_save_img(series,f,fi+len(func),i))


fig=plt.figure(figsize=(10, 10))
columns = len(df)
rows = len(mod)+len(org)


for i,img in imgs :
    print(i)
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img)
plt.show()
