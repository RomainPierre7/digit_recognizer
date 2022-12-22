import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from skimage.transform import resize
from PIL import Image
import tkinter

data = pd.read_csv(r"data/train.csv",encoding = 'ISO-8859-1', index_col=0)
start_col = 0
n = data.shape[0]
p=data.shape[1] - start_col
X=data.loc[:, data.columns[start_col]:data.columns[start_col + p - 1]].to_numpy().reshape([n,p])

d = 30000

train = X[:d]
train_target = data.index[:d]
test = X[d:]
test_target = data.index[d:]

def im_array(L):
    IM = []
    for i in range(28):
        x = []
        for j in range(28):
            x.append(L[28*i + j])
        IM.append(x)
    return IM

mlp=MLPClassifier(solver='adam', hidden_layer_sizes=500, alpha=1e-06, verbose=True)

print("Please wait, model training in progress...")

mlp.fit(train,train_target)

prediction = mlp.predict(test)

print("Model accuracy :", accuracy_score(test_target, prediction))
window=0
canvas=0

lastX=0
lastY=0

couleur='black'

def click(event):
    global lastX, lastY
    lastX=event.x
    lastY=event.y

def move(event):
    global lastX, lastY
    canvas.create_line(lastX, lastY, event.x, event.y, fill=couleur, width=50)
    lastX=event.x
    lastY=event.y

def init_window():
    global window, canvas
    window=tkinter.Tk()
    window.title('Digit recongnizer')
    window.state('normal')
    canvas = tkinter.Canvas(window, width=1000, height=1000, bg='white')
    canvas.bind('<Button-1>', click)
    canvas.bind('<B1-Motion>', move)
    canvas.grid(row=0, column=0, rowspan=4)

def clear():
    canvas.delete(tkinter.ALL)

def save_as_png():
    canvas.postscript(file = r'data/digit' + '.eps')
    img = Image.open(r'data/digit' + '.eps')
    img.save(r'data/digit' + '.png', 'png')

def predict():
    save_as_png()
    im = plt.imread(r'data/digit.png')
    res = resize(im, (28,28))
    res = res[:,:,2].reshape(784,)
    for i in range(784):
        res[i] = int(abs(res[i] - 1)*255)
    predict = mlp.predict(res.reshape(1,784))
    label2["text"]="%3s" %predict[0]

init_window()
btn1=tkinter.Button(window, text="submit", command=predict)
btn1.grid(row=0, column=1, sticky=tkinter.N)
btn2=tkinter.Button(window, text="clear", command=clear)
btn2.grid(row=0, column=2, sticky=tkinter.N)
label1=tkinter.Label(window, text="Digit :", bg='red', width=15)
label1.grid(row=1, column=1, sticky=tkinter.N)
label2=tkinter.Label(window, text="%3s" %0, width=15)
label2.grid(row=1, column=2, sticky=tkinter.N)
window.mainloop()