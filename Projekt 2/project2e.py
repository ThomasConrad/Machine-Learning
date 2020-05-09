import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
import torch

def train_network(x, y, model):
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    nnx = torch.as_tensor(x, dtype=torch.float)
    nny = torch.as_tensor(y, dtype=torch.float)
    
    for t in range(1000):
        y_prediction = model(nnx)
    
        loss = loss_fn(y_prediction.squeeze(), nny)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def get_generalization_error_ann(x, y, hidden_units):
    K = 5
    kf = KFold(n_splits=K)
    generalization_error = 0

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        nnx = torch.as_tensor(x_train, dtype=torch.float)
        nny = torch.as_tensor(y_train, dtype=torch.float)

        model = torch.nn.Sequential(
        torch.nn.Linear(8,hidden_units),
        torch.nn.Sigmoid(),
        torch.nn.Linear(hidden_units,1),
        torch.nn.Sigmoid()
        )

        train_network(nnx, nny, model)

        #Sums amount of correctly predicted, divides by total amount
        temp=(model(torch.as_tensor(x_test,dtype=torch.float))>=0.5).squeeze().numpy()
        test_error_nn = (np.size(y_test)-np.sum(y_test==temp)) / np.size(y_test)

        generalization_error += len(y_test)/len(y)*test_error_nn

    return generalization_error

def get_generalization_error_glm(x, y, model):
    K = 5
    kf = KFold(n_splits=K)
    generalization_error = 0

    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
        model.fit(x_train,y_train)
    
        test_error_glm = (np.size(y_test)-np.sum(y_test==(reg.predict(x_test)>=0.5))) / np.size(y_test)

        generalization_error += len(y_test)/len(y)*test_error_glm

    return generalization_error



data = np.loadtxt("prostate.data",delimiter="\t",skiprows=1,usecols=[1,2,3,4,5,6,7,8,9])

np.random.seed(123)
np.random.shuffle(data)

m = np.mean(data,axis=0)
sd = np.std(data,axis=0)
norm_data = (data-m)/sd
y = data[:,4]
x = np.concatenate((norm_data[:,:4],norm_data[:,5:]),axis=1)



lambdas = np.power(10,np.linspace(-5,5,101))

hidden_units = [1,2,3,4,5,6,7,8,9,10]


x_test = x[0:int(np.floor(len(x)/3))]
x_train = x[int(np.floor(len(x)/3)):]

y_test = y[0:int(np.floor(len(y)/3))]
y_train = y[int(np.floor(len(y)/3)):]


#BASELINE MODEL
baseline = y_test.mean()

#LINEAR REGRESSION MODEL
generalization_errors_list_glm = []
for i in lambdas:
    reg = lm.LogisticRegression(C=1/i)
    generalization_errors_list_glm.append(get_generalization_error_glm(x_train, y_train, reg))
    optimal_lambda = lambdas[np.argmin(generalization_errors_list_glm)]

reg = lm.LogisticRegression(C=1/optimal_lambda)
reg.fit(x_train,y_train)

#ANN MODEL
generalization_errors_list_nn = []
for i in hidden_units:
    generalization_errors_list_nn.append(get_generalization_error_ann(x_train, y_train, i))
    optimal_hu = int(hidden_units[np.argmin(generalization_errors_list_nn)])

nnx = torch.as_tensor(x_train, dtype=torch.float)
nny = torch.as_tensor(y_train, dtype=torch.float)

ANN = torch.nn.Sequential(
    torch.nn.Linear(8,optimal_hu),
    torch.nn.Sigmoid(),
    torch.nn.Linear(optimal_hu,1),
    torch.nn.Sigmoid()
    )
train_network(x_train, y_train, ANN)

print(optimal_lambda,optimal_hu)



temp=reg.predict(x_test)
reg_confusion = [[np.sum(temp*y_test),np.sum(temp*(1-y_test))],[np.sum((1-temp)*y_test),np.sum((1-temp)*(1-y_test))]]
temp=((ANN(torch.as_tensor(x_test,dtype=torch.float))>=0.5).squeeze().numpy())
ann_confusion = [[np.sum(temp*y_test),np.sum(temp*(1-y_test))],[np.sum((1-temp)*y_test),np.sum((1-temp)*(1-y_test))]]
temp = 0
base_confusion= [[np.sum(temp*y_test),np.sum(temp*(1-y_test))],[np.sum((1-temp)*y_test),np.sum((1-temp)*(1-y_test))]]


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

plt.figure(1)
df_cm = pd.DataFrame(reg_confusion, ["true","false"], ["true","false"])
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16},cmap='binary',cbar=False, linewidths=1, linecolor='black')# font size
plt.xlabel('Ground truth')
plt.ylabel('Predicted value')

plt.figure(2)
df_cm = pd.DataFrame(ann_confusion, ["true","false"], ["true","false"])
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16},cmap='binary',cbar=False, linewidths=1, linecolor='black')# font size
plt.xlabel('Ground truth')
plt.ylabel('Predicted value')

plt.figure(3)
df_cm = pd.DataFrame(base_confusion, ["true","false"], ["true","false"])
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16},cmap='binary',cbar=False, linewidths=1, linecolor='black')# font size
plt.xlabel('Ground truth')
plt.ylabel('Predicted value')

plt.show()

