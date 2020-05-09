import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from scipy import stats
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
    K = 10
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
    K = 10
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


x_test = x[0:int(np.floor(len(x)/3))]
x_train = x[int(np.floor(len(x)/3)):]

y_test = y[0:int(np.floor(len(y)/3))]
y_train = y[int(np.floor(len(y)/3)):]

#BASELINE MODEL
baseline = y_test.mean()

#LINEAR REGRESSION MODEL
generalization_errors_list_glm = []
for i in lambdas:
    reg = lm.Ridge(alpha=i)
    generalization_errors_list_glm.append(get_generalization_error_glm(x, y, reg))
    optimal_lambda = lambdas[np.argmin(generalization_errors_list_glm)]

reg = lm.Ridge(alpha=optimal_lambda)
reg.fit(x_train, y_train)

#ANN MODEL
generalization_errors_list_nn = []
for i in hidden_units:
    generalization_errors_list_nn.append(get_generalization_error_ann(x, y, i))
    optimal_hu = int(hidden_units[np.argmin(generalization_errors_list_nn)])

ANN = torch.nn.Sequential(
        torch.nn.Linear(8,optimal_hu),
        torch.nn.Sigmoid(),
        torch.nn.Linear(optimal_hu,1)
        )
train_network(x_train, y_train, ANN)

print(optimal_lambda,optimal_hu)



#One-liner that calculates the pairwise differences in the losses of the Linear model and the ANN model
z = ((y_test-reg.predict(x_test))**2-(y_test-ANN(torch.as_tensor(x_test,dtype=torch.float)).reshape(1,-1).detach().numpy().squeeze())**2)
zm = z.mean()
zsd = z.std()
t_obs = zm/(zsd/np.sqrt(len(y_test)))
print(zm)

p_reg_ann = 2*stats.t.cdf(-abs(t_obs), df=len(y_test)-1)

#One-liner that calculates the pairwise differences in the losses of the Linear model and the baseline
z = ((y_test-reg.predict(x_test))**2-(y_test-baseline)**2)
zm = z.mean()
zsd = z.std()
t_obs = zm/(zsd/np.sqrt(len(y_test)))
print(zm)

p_reg_base = 2*stats.t.cdf(-abs(t_obs), df=len(y_test)-1)

#One-liner that calculates the pairwise differences in the losses of the ANN model and the baseline
z = ((y_test-ANN(torch.as_tensor(x_test,dtype=torch.float)).reshape(1,-1).detach().numpy().squeeze())**2-(y_test-baseline)**2)
zm = z.mean()
zsd = z.std()
t_obs = zm/(zsd/np.sqrt(len(y_test)))
print(zm)

p_ann_base = 2*stats.t.cdf(-abs(t_obs), df=len(y_test)-1)

print(p_reg_ann)
print(p_reg_base)
print(p_ann_base)



