import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.model_selection import KFold

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

def get_generalization_error(x, y, model):
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

generalization_errors_list = []

y = norm_data[:,0]
x = norm_data[:,1:]


lambdas = np.power(10,np.linspace(-5,5,101))
generalization_errors_list = []

for i in lambdas:
    reg = lm.Ridge(alpha=i)
    generalization_errors_list.append(get_generalization_error(x, y, reg))



plt.semilogx(lambdas,generalization_errors_list)
plt.ylabel('generalization error', fontdict=font)
plt.xlabel('Regularization parameter', fontdict=font)
plt.show()

print("Lambda =", lambdas[np.argmin(generalization_errors_list)])