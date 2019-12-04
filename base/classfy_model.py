from base.base import BaseExecutor
import numpy as np
from sklearn.metrics import roc_auc_score


class ClassfyModel(BaseExecutor):

    def __init__(self, **kwargs):
        self.data = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.pre_data = None
        self.pre_proba = None
        self.y_ = None
        self.coef_ = None
        self.intercept_ = None

    def load_data(self) -> tuple:
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self) -> np.array:
        raise NotImplementedError

    # 尝试自己实现auc算法
    def get_auc(self):
        # 按照得分降序排列
        _roc_data = zip(self.pre_proba, self.y_test)
        roc_data = sorted(_roc_data, key=lambda x: x[0])
        # 正标签的rank求和，如果出现得分一样的，需要取平均，比如排名2和3的分数一样，其中一个是正标签，则rank值为2.5:(2 + 3) / 2
        sum_rank = 0
        # 上一条数据的得分和标签，初始化为第一条数据的得分和标签
        last_prob, last_label = roc_data[0]
        # 当出现相同得分时，需要临时存储当前相同得分的rank总量，最后再求平均
        tmp_sum_rank = 1
        # 存储当前得分相同的个数
        tmp_same_cnt = 1
        # 正标签样本数
        true_cnt = tmp_true_cnt = last_label
        # 负标签样本数
        false_cnt = tmp_false_cnt = 1 - last_label

        # 遍历计算，从第2项开始，注意
        for i in range(1, len(roc_data)):
            # 当前记录的得分和标签
            curr_prob, curr_label = roc_data[i]
            # 更新正负样本的总量
            true_cnt += curr_label
            false_cnt += (1 - curr_label)
            # 如果当前记录和上条记录得分不同，则计算sum_rank，并重新初始化tmp_same_cnt，tmp_sum_rank，tmp_true_cnt，tmp_false_cnt
            if curr_prob != last_prob:
                sum_rank += (tmp_sum_rank / tmp_same_cnt) * tmp_true_cnt
                tmp_same_cnt = 1
                tmp_sum_rank = i + 1
                tmp_true_cnt = curr_label
                tmp_false_cnt = 1 - curr_label
            # 如果当前记录和上条记录得分相同，则累计更新tmp_same_cnt，tmp_sum_rank，tmp_true_cnt，tmp_false_cnt
            else:
                tmp_same_cnt += 1
                tmp_sum_rank += i + 1
                tmp_true_cnt += curr_label
                tmp_false_cnt += (1 - curr_label)
            # 将本条记录更新为上一条记录
            last_label, last_prob = curr_label, curr_prob
        sum_rank += (tmp_sum_rank / tmp_same_cnt) * tmp_true_cnt
        auc_rate = (sum_rank - true_cnt * (true_cnt + 1) / 2) / (true_cnt * false_cnt)
        return auc_rate

    def get_sklearn_auc(self):
        auc_rate = roc_auc_score(self.y_test, self.pre_proba)
        return auc_rate

    # 几个评估模型的指标
    def estimate_model(self):
        assert (self.pre_data is not None), f'self.pre_data为None，模型尚未预测，请尝试先执行predict方法加载预测数据'
        print(self.__class__.__name__, 'auc分数', self.get_auc())
        print(self.__class__.__name__, 'sklearn auc分数', self.get_sklearn_auc())
