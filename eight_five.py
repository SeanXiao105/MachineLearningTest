import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.utils import resample
def stumpClassify(X, dim, thresh_val, thresh_inequal):
    ret_array = np.ones((X.shape[0], 1))

    if thresh_inequal == 'lt':
        ret_array[X[:, dim] <= thresh_val] = -1
    else:
        ret_array[X[:, dim] > thresh_val] = -1

    return ret_array
#建立树桩
def buildStump(X, y):
    m, n = X.shape
    best_stump = {}
    min_error = 1
    for dim in range(n):
        x_min = np.min(X[:, dim])
        x_max = np.max(X[:, dim])
        # 这里第一次尝试使用排序后的点作为分割点，效果很差，因为那样会错过一些更好的分割点；
        # 所以后来切割点改成将最大值和最小值之间分割成20等份。
        split_points = [(float(x_max) - float(x_min)) / 20 * i + x_min for i in range(20)]
        for inequal in ['lt', 'gt']:
            for thresh_val in split_points:
                ret_array = stumpClassify(X, dim, thresh_val, inequal)
                error = np.mean(ret_array != y)
                if error < min_error:
                    best_stump['dim'] = dim
                    best_stump['thresh'] = thresh_val
                    best_stump['inequal'] = inequal
                    best_stump['error'] = error
                    min_error = error
    return best_stump
def stumpBagging(X, y, nums=20):
    stumps = []
    seed = 16
    for _ in range(nums):
        X_, y_ = resample(X, y, random_state=seed)  # sklearn 中自带的实现自助采样的方法
        seed += 1
        stumps.append(buildStump(X_, y_))
    return stumps
def stumpPredict(X, stumps):
    ret_arrays = np.ones((X.shape[0], len(stumps)))

    for i, stump in enumerate(stumps):
        ret_arrays[:, [i]] = stumpClassify(X, stump['dim'], stump['thresh'], stump['inequal'])

    return np.sign(np.sum(ret_arrays, axis=1))
#可视化
def pltStumpBaggingDecisionBound(X_, y_, stumps):
    pos = y_ == 1
    neg = y_ == -1
    x_tmp = np.linspace(0, 1, 600)
    y_tmp = np.linspace(-0.1, 0.7, 600)

    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)
    Z_ = stumpPredict(np.c_[X_tmp.ravel(), Y_tmp.ravel()], stumps).reshape(X_tmp.shape)

    plt.contour(X_tmp, Y_tmp, Z_, [0], colors='orange', linewidths=1)

    plt.scatter(X_[pos, 0], X_[pos, 1], label='好瓜', color='c')
    plt.scatter(X_[neg, 0], X_[neg, 1], label='坏瓜', color='lightcoral')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
   # plt.legend(loc='upper left')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import numpy as np

    #     import pandas as pd
    #     data_path = 'F:\\python\\dataset\\chapter8\\watermelon3_0a_Ch.txt'

    #     data = pd.read_table(data_path, delimiter=' ')

    #     X = data.iloc[:, :2].values
    #     y = data.iloc[:, 2].values
    #     XX = np.array(X)
    #     yy = np.array(y)
    XX = [[0.697, 0.460],
          [0.774, 0.376],
          [0.634, 0.264],
          [0.608, 0.318],
          [0.556, 0.215],
          [0.403, 0.237],
          [0.481, 0.149],
          [0.437, 0.211],
          [0.666, 0.091],
          [0.243, 0.267],
          [0.245, 0.057],
          [0.343, 0.099],
          [0.639, 0.161],
          [0.657, 0.198],
          [0.360, 0.370],
          [0.593, 0.042],
          [0.719, 0.103]
          ]
    yy = [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    X = np.array(XX)
    y = np.array(yy)
    stumps = stumpBagging(X, y, 21)

    print(np.mean(stumpPredict(X, stumps) == y))
    pltStumpBaggingDecisionBound(X, y, stumps)

