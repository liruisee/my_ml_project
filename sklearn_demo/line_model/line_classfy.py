import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from base.classfy_model import ClassfyModel
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


class LoadDataModel(ClassfyModel):

    def __init__(self):
        super().__init__()
        self.init_model = LinearDiscriminantAnalysis()

    # 加载数据，选择了波士顿的房价数据，并采用留出法划分训练集和测试集
    def load_data(self):
        if self.data is None:
            self.data = load_breast_cancer()
        x, y = self.data.data, self.data.target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y)

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearDiscriminantAnalysisModel(LoadDataModel):

    def __init__(self):
        super().__init__()
        self.init_model = LinearDiscriminantAnalysis()

    def train(self):
        assert (self.x_train is not None and self.y_train is not None), '训练数据为None，无法进行训练，请尝试先执行load_data方法'
        self.init_model.fit(self.x_train, self.y_train)
        self.model = self.init_model
        return self.init_model

    def predict(self):
        pre_datas = []
        probas = []
        assert (self.model is not None), f'self.model值为None，无法进行预测'
        pre_proba_tuple = self.model.predict_proba(self.x_test)
        assert(len(pre_proba_tuple[0]) == 2), f'当前模型仅处理二分类，当前分类元祖为：{pre_proba_tuple[0]}'
        for tp in pre_proba_tuple:
            label = np.argmax(tp)
            proba = tp[label]
            pre_datas.append(label)
            probas.append(proba)
        self.pre_data = np.array(pre_datas)
        self.pre_proba = np.array(probas)
        return self.pre_data


class SVCModel(LoadDataModel):

    def __init__(self):
        super().__init__()
        self.init_model = SVC(probability=True)

    def train(self):
        assert (self.x_train is not None and self.y_train is not None), '训练数据为None，无法进行训练，请尝试先执行load_data方法'
        self.init_model.fit(self.x_train, self.y_train)
        self.model = self.init_model
        return self.init_model

    def predict(self):
        pre_datas = []
        probas = []
        assert (self.model is not None), f'self.model值为None，无法进行预测'
        pre_proba_tuple = self.model.predict_proba(self.x_test)
        assert(len(pre_proba_tuple[0]) == 2), f'当前模型仅处理二分类，当前分类元祖为：{pre_proba_tuple[0]}'
        for tp in pre_proba_tuple:
            label = np.argmax(tp)
            proba = tp[label]
            pre_datas.append(label)
            probas.append(proba)
        self.pre_data = np.array(pre_datas)
        self.pre_proba = np.array(probas)
        return self.pre_data


class SGDClassifierModel(LoadDataModel):

    def __init__(self):
        super().__init__()
        self.init_model = SGDClassifier(loss="log", penalty="l2")

    def train(self):
        assert (self.x_train is not None and self.y_train is not None), '训练数据为None，无法进行训练，请尝试先执行load_data方法'
        self.init_model.fit(self.x_train, self.y_train)
        self.model = self.init_model
        return self.init_model

    def predict(self):
        pre_datas = []
        probas = []
        assert (self.model is not None), f'self.model值为None，无法进行预测'
        pre_proba_tuple = self.model.predict_proba(self.x_test)
        assert(len(pre_proba_tuple[0]) == 2), f'当前模型仅处理二分类，当前分类元祖为：{pre_proba_tuple[0]}'
        for tp in pre_proba_tuple:
            label = np.argmax(tp)
            proba = tp[label]
            pre_datas.append(label)
            probas.append(proba)
        self.pre_data = np.array(pre_datas)
        self.pre_proba = np.array(probas)
        return self.pre_data
