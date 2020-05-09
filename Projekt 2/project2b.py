import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
import torch

def train_network(x, y, model):
    loss_fn = torch.nn.MSELoss()
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
        torch.nn.Linear(hidden_units,1)
        )

        train_network(nnx, nny, model)

        tx = torch.as_tensor(x_test, dtype=torch.float)
        ty = torch.as_tensor(y_test, dtype=torch.float)

        #Subtracts y_test from predicted y, then divides by length of y_test
        test_error_nn = (((model(tx).reshape(1,-1).squeeze()-ty)**2).sum()/ty.size(0)).item()
    
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
    
        test_error = 0
        for i in range(len(y_test)):
            test_error += (y_test[i]-model.predict([x_test[i]]))**2
        test_error *= 1/len(y_test)
    
        generalization_error += len(y_test)/len(y)*test_error

    return generalization_error



data = np.loadtxt("prostate.data",delimiter="\t",skiprows=1,usecols=[1,2,3,4,5,6,7,8,9])
m = np.mean(data,axis=0)
sd = np.std(data,axis=0)
norm_data = (data-m)/sd
y = norm_data[:,0]
x = norm_data[:,1:]






lambdas = np.power(10,np.linspace(-5,5,101))

hidden_units = [1,2,3,4,5,6,7,8,9,10]

K = 5
kf = KFold(n_splits=K)

k = 1
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #BASELINE (MEAN)
    test_error_base = y_test.var()

    #LINEAR REGRESSION MODEL
    generalization_errors_list_glm = []
    for i in lambdas:
        reg = lm.Ridge(alpha=i)
        generalization_errors_list_glm.append(get_generalization_error_glm(x_train, y_train, reg))
        optimal_lambda = lambdas[np.argmin(generalization_errors_list_glm)]

    model = lm.Ridge(alpha=optimal_lambda)
    model.fit(x_train,y_train)

    test_error_glm = 0
    for i in range(len(y_test)):
        test_error_glm += (y_test[i]-model.predict([x_test[i]]))**2
    test_error_glm *= 1/len(y_test)
    test_error_glm = test_error_glm[0]

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
        torch.nn.Linear(optimal_hu,1)
        )

    train_network(nnx, nny, ANN)

    tx = torch.as_tensor(x_test, dtype=torch.float)
    ty = torch.as_tensor(y_test, dtype=torch.float)

    #Subtracts y_test from predicted y, then divides by length of y_test
    test_error_nn = (((ANN(tx).reshape(1,-1).squeeze()-ty)**2).sum()/ty.size(0)).item()


    print("k:",k,"Regression Error:",test_error_glm,"Lambda:",optimal_lambda,"ANN Error",test_error_nn,"Hidden Units:",optimal_hu,"Baseline:",test_error_base)
    k += 1