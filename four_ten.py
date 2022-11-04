import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from decision_tree import treePlottter


class Node(object):
    def __init__(self):
        self.feature_name = None  # 特性的名称
        self.feature_index = None  # 特性的下标
        self.subtree = {}  # 树节点的集合
        self.impurity = None  # 信息此节点的信息增益
        self.is_continuous = False  # 是否为连续值
        self.split_value = None  # 连续值时的划分依据
        self.is_leaf = False  # 是否为叶子节点
        self.leaf_class = None  # 叶子节点对应的类
        self.leaf_num = 0  # 叶子数目
        self.high = -1  # 树的高度


def entroy(y):
    p = pd.value_counts(y) / y.shape[0]  # 计算各类样本所占比率
    ent = np.sum(-p * np.log2(p))
    return ent

    return node


def gini(y):
    p = pd.value_counts(y) / y.shape[0]
    gini = 1 - np.sum(p ** 2)
    return gini


def gini_index(feature, y, is_continuous=False):
    """
    计算基尼指数， 对于连续值，选择基尼系统最小的点，作为分割点
    -------
    :param feature:
    :param y:
    :return:
    """
    m = y.shape[0]
    unique_value = pd.unique(feature)
    if is_continuous:
        unique_value.sort()  # 排序, 用于建立分割点
        # 这里其实也可以直接用feature值作为分割点，但这样会出现空集， 所以还是按照书中4.7式建立分割点。好处是不会出现空集
        split_point_set = [(unique_value[i] + unique_value[i + 1]) / 2 for i in range(len(unique_value) - 1)]

        min_gini = float('inf')
        min_gini_point = None
        for split_point_ in split_point_set:  # 遍历所有的分割点，寻找基尼指数最小的分割点
            Dv1 = y[feature <= split_point_]
            Dv2 = y[feature > split_point_]
            gini_index = Dv1.shape[0] / m * gini(Dv1) + Dv2.shape[0] / m * gini(Dv2)

            if gini_index < min_gini:
                min_gini = gini_index
                min_gini_point = split_point_
        return [min_gini, min_gini_point]
    else:
        gini_index = 0
        for value in unique_value:
            Dv = y[feature == value]
            m_dv = Dv.shape[0]
            gini_ = gini(Dv)  # 原书4.5式
            gini_index += m_dv / m * gini_  # 4.6式

        return [gini_index]


def info_gain(feature, y, entD, is_continuous=False):
    """
    计算信息增益
    ------
    :param feature: 当前特征下所有样本值
    :param y:       对应标签值
    :return:        当前特征的信息增益, list类型，若当前特征为离散值则只有一个元素为信息增益，若为连续值，则第一个元素为信息增益，第二个元素为切分点
    """
    m = y.shape[0]
    unique_value = pd.unique(feature)
    if is_continuous:
        unique_value.sort()  # 排序, 用于建立分割点
        split_point_set = [(unique_value[i] + unique_value[i + 1]) / 2 for i in range(len(unique_value) - 1)]
        min_ent = float('inf')  # 挑选信息熵最小的分割点
        min_ent_point = None
        for split_point_ in split_point_set:

            Dv1 = y[feature <= split_point_]
            Dv2 = y[feature > split_point_]
            feature_ent_ = Dv1.shape[0] / m * entroy(Dv1) + Dv2.shape[0] / m * entroy(Dv2)

            if feature_ent_ < min_ent:
                min_ent = feature_ent_
                min_ent_point = split_point_
        gain = entD - min_ent

        return [gain, min_ent_point]

    else:
        feature_ent = 0
        for value in unique_value:
            Dv = y[feature == value]  # 当前特征中取值为 value 的样本，即书中的 D^{v}
            feature_ent += Dv.shape[0] / m * entroy(Dv)

        gain = entD - feature_ent  # 原书中4.2式
        return [gain]


def info_gainRatio(feature, y, entD, is_continuous=False):
    """
    计算信息增益率 参数和info_gain方法中参数一致
    ------
    :param feature:
    :param y:
    :param entD:
    :return:
    """

    if is_continuous:
        # 对于连续值，以最大化信息增益选择划分点之后，计算信息增益率，注意，在选择划分点之后，需要对信息增益进行修正，要减去log_2(N-1)/|D|，N是当前特征的取值个数，D是总数据量。
        # 修正原因是因为：当离散属性和连续属性并存时，C4.5算法倾向于选择连续特征做最佳树分裂点
        # 信息增益修正中，N的值，网上有些资料认为是“可能分裂点的个数”，也有的是“当前特征的取值个数”，这里采用“当前特征的取值个数”。
        # 这样 (N-1)的值，就是去重后的“分裂点的个数” , 即在info_gain函数中，split_point_set的长度，个人感觉这样更加合理。有时间再看看原论文吧。

        gain, split_point = info_gain(feature, y, entD, is_continuous)
        p1 = np.sum(feature <= split_point) / feature.shape[0]  # 小于或划分点的样本占比
        p2 = 1 - p1  # 大于划分点样本占比
        IV = -(p1 * np.log2(p1) + p2 * np.log2(p2))

        grain_ratio = (gain - np.log2(feature.nunique()) / len(y)) / IV  # 对信息增益修正
        return [grain_ratio, split_point]
    else:
        p = pd.value_counts(feature) / feature.shape[0]  # 当前特征下 各取值样本所占比率
        IV = np.sum(-p * np.log2(p))  # 原书4.4式
        grain_ratio = info_gain(feature, y, entD, is_continuous)[0] / IV
        return [grain_ratio]


def choose_best_feature_gini(X, y):
    features = X.columns
    best_feature_name = None
    best_gini = [float('inf')]
    for feature_name in features:
        is_continuous = type_of_target(X[feature_name]) == 'continuous'
        gini_idex = gini_index(X[feature_name], y, is_continuous)
        if gini_idex[0] < best_gini[0]:
            best_feature_name = feature_name
            best_gini = gini_idex

    return best_feature_name, best_gini


def choose_best_feature_gainratio(X, y):
    """
    以返回值中best_gain_ratio 的长度来判断当前特征是否为连续值，若长度为 1 则为离散值，若长度为 2 ， 则为连续值
    :param X: 当前所有特征的数据 pd.DaraFrame格式
    :param y: 标签值
    :return:  以信息增益率来选择的最佳划分属性，第一个返回值为属性名称，第二个为最佳划分属性对应的信息增益率
    """
    features = X.columns
    best_feature_name = None

    best_gain_ratio = [float('-inf')]
    entD = entroy(y)

    for feature_name in features:
        is_continuous = type_of_target(X[feature_name]) == 'continuous'
        info_gain_ratio = info_gainRatio(X[feature_name], y, entD, is_continuous)
        if info_gain_ratio[0] > best_gain_ratio[0]:
            best_feature_name = feature_name
            best_gain_ratio = info_gain_ratio

    return best_feature_name, best_gain_ratio


def choose_best_feature_infogain(X, y):
    """
    以返回值中best_info_gain 的长度来判断当前特征是否为连续值，若长度为 1 则为离散值，若长度为 2 ， 则为连续值
    :param X: 当前所有特征的数据 pd.DaraFrame格式
    :param y: 标签值
    :return:  以信息增益来选择的最佳划分属性，第一个返回值为属性名称，
    """
    features = X.columns
    best_feature_name = None
    best_info_gain = [float('-inf')]
    entD = entroy(y)
    for feature_name in features:
        is_continuous = type_of_target(X[feature_name]) == 'continuous'
        infogain = info_gain(X[feature_name], y, entD, is_continuous)
        if infogain[0] > best_info_gain[0]:
            best_feature_name = feature_name
            best_info_gain = infogain

    return best_feature_name, best_info_gain


def generate(X, y, columns, criterion):
    node = Node()
    # Pandas.Series.nunique()统计不同值的个数
    if y.nunique() == 1:  # 属于同一类别
        node.is_leaf = True
        node.leaf_class = y.values[0]
        node.high = 0
        node.leaf_num += 1
        return node

    if X.empty:  # 特征用完了，数据为空，返回样本数最多的类
        node.is_leaf = True
        node.leaf_class = pd.value_counts(y).index[0]  # 返回样本数最多的类
        node.high = 0
        node.leaf_num += 1
        return node

    if criterion == 'gini':
        best_feature_name, best_impurity = choose_best_feature_gini(X, y)
    elif criterion == 'infogain':
        best_feature_name, best_impurity = choose_best_feature_infogain(X, y)
    elif criterion == 'gainratio':
        best_feature_name, best_impurity = choose_best_feature_gainratio(X, y)
    # best_feature_name, best_impurity = choose_best_feature_infogain(X, y)

    node.feature_name = best_feature_name
    node.impurity = best_impurity[0]
    node.feature_index = columns.index(best_feature_name)

    feature_values = X.loc[:, best_feature_name]

    if len(best_impurity) == 1:  # 离散值
        node.is_continuous = False

        unique_vals = pd.unique(feature_values)
        sub_X = X.drop(best_feature_name, axis=1)

        max_high = -1
        for value in unique_vals:
            node.subtree[value] = generate(sub_X[feature_values == value], y[feature_values == value], columns,
                                           criterion)
            if node.subtree[value].high > max_high:  # 记录子树下最高的高度
                max_high = node.subtree[value].high
            node.leaf_num += node.subtree[value].leaf_num

        node.high = max_high + 1

    elif len(best_impurity) == 2:  # 连续值
        node.is_continuous = True
        node.split_value = best_impurity[1]
        up_part = '>= {:.3f}'.format(node.split_value)
        down_part = '< {:.3f}'.format(node.split_value)

        node.subtree[up_part] = generate(X[feature_values >= node.split_value],
                                         y[feature_values >= node.split_value], columns, criterion)
        node.subtree[down_part] = generate(X[feature_values < node.split_value],
                                           y[feature_values < node.split_value], columns, criterion)

        node.leaf_num += (node.subtree[up_part].leaf_num + node.subtree[down_part].leaf_num)

        node.high = max(node.subtree[up_part].high, node.subtree[down_part].high) + 1

    return node


if __name__ == "__main__":
    data = pd.read_csv("西瓜3.0.txt", index_col=0)  # index_col参数设置第一列作为index
    # 不带第一列，求得西瓜的属性
    x = data.iloc[:, :8]  # <class 'pandas.core.frame.DataFrame'>
    y = data.iloc[:, 8]  # <class 'pandas.core.series.Series'>
    columns_name = list(x.columns)  # 包括原数据的列名
    criterion = "gini"  # 'gainratio''infogain': 'gini':
    node = generate(x, y, columns_name, criterion)
    treePlottter.create_plot(node)
