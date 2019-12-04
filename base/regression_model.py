from base.base import BaseExecutor

__all__ = ['RegressionModel']


class RegressionModel(BaseExecutor):

    def __init__(self, **kwargs):
        self.data = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.pre_data = None
        self.y_ = None
        self.coef_ = None
        self.intercept_ = None

    # 训练模型，采用线性回归
    def load_data(self):
        raise NotImplementedError

    # 训练模型，采用线性回归
    def train(self):
        raise NotImplementedError

    # 预测模型
    def predict(self):
        raise NotImplementedError

    # 计算y均值
    def get_y_(self):
        if self.y_ is not None:
            return self.y_
        # 所有y值求和，为了后续取平均
        sum_y = 0
        # 测试样本总数
        _cnt = len(self.y_test)
        for i in range(_cnt):
            sum_y += self.y_test[i]
        # 样本标签均值
        self.y_ = sum_y / _cnt
        return self.y_

    # 计算r方
    def get_r2_scrose(self):
        assert (self.pre_data is not None), f'self.pre_data为None，模型尚未预测，请尝试先执行predict方法加载预测数据'
        # 样本标签均值
        y_ = self.get_y_()
        # 分子
        numerator = 0
        # 分母
        denominator = 0
        # 测试样本数
        _cnt = len(self.y_test)
        # 计算r方分数
        for i in range(_cnt):
            numerator += ((self.y_test[i] - self.pre_data[i]) ** 2)
            denominator += ((self.y_test[i] - y_) ** 2)
        return 1 - (numerator / denominator)

    # 平均绝对误差
    def get_mae(self):
        assert (self.pre_data is not None), f'self.pre_data为None，模型尚未预测，请尝试先执行predict方法加载预测数据'
        _cnt = len(self.y_test)
        sum_err = 0
        for i in range(_cnt):
            sum_err += abs(self.y_test[i] - self.pre_data[i])
        return sum_err / _cnt

    # 均方误差
    def get_mse(self):
        assert (self.pre_data is not None), f'self.pre_data为None，模型尚未预测，请尝试先执行predict方法加载预测数据'
        _cnt = len(self.y_test)
        sum_err = 0
        for i in range(_cnt):
            sum_err += (self.y_test[i] - self.pre_data[i]) ** 2
        return sum_err / _cnt

    # 均方根误差
    def get_rmse(self):
        import math
        assert (self.pre_data is not None), f'self.pre_data为None，模型尚未预测，请尝试先执行predict方法加载预测数据'
        _cnt = len(self.y_test)
        sum_err = 0
        for i in range(_cnt):
            sum_err += (self.y_test[i] - self.pre_data[i]) ** 2
        return math.sqrt(sum_err / _cnt)

    # 均方根误差
    def get_mape(self):
        assert (self.pre_data is not None), f'self.pre_data为None，模型尚未预测，请尝试先执行predict方法加载预测数据'
        _cnt = len(self.y_test)
        sum_err = 0
        for i in range(_cnt):
            sum_err += abs((self.y_test[i] - self.pre_data[i]) / self.y_test[i])
        return sum_err / _cnt * 100

    # 几个评估模型的指标
    def estimate_model(self):
        assert (self.pre_data is not None), f'self.pre_data为None，模型尚未预测，请尝试先执行predict方法加载预测数据'
        print(self.__class__.__name__, '平均绝对误差', self.get_mae())
        print(self.__class__.__name__, '平均平方误差', self.get_mse())
        print(self.__class__.__name__, '平均平方根误差', self.get_rmse())
        print(self.__class__.__name__, '平均绝对百分比误差', self.get_mape())
        print(self.__class__.__name__, 'r方分数', self.get_r2_scrose())
