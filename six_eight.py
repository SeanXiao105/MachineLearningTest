import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

def set_ax_gray(ax):
    ax.patch.set_facecolor("gray")
    ax.patch.set_alpha(0.1)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.grid(axis='y',linestyle='-.')

data=pd.read_table(r"西瓜3.0a.txt",delimiter=' ',dtype=float)
x=data.iloc[:,[0]].values
y=data.iloc[:,1].values
gamma=10
C=1
ax=plt.subplot()
set_ax_gray(ax)
ax.scatter(x,y,color='c',label='data')
for gamma in [1,10,100,1000]:
    svr=svm.SVR(kernel='rbf',gamma=gamma,C=C)
    svr.fit(x,y)
    ax.plot(np.linspace(0.2,0.8),svr.predict(np.linspace(0.2,0.8).reshape(-1,1)),
            label='gamma={},C={}'.format(gamma,C))

ax.legend(loc='upper left')
ax.set_xlabel('密度')
ax.set_ylabel('含糖率')
plt.show()