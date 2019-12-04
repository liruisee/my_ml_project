from sklearn.model_selection import train_test_split
from sklearn import datasets
import lightgbm as lgb
import numpy as np


# lightgbm分类数，预测数据为连续性，预测标签为0~9
class LgbDigits:
    # 初始化模型文件路径
    def __init__(self, data=None):
        # 加载数据
        self.data = data
        if self.data is not None:
            assert (('data' in data) and ('target' in data)), 'data中必须包含{"data", "target"}两个key'
        # 模型对象
        self.model = None
        # x_train, x_test, y_train, y_test
        self.data_tuple = None
        # 模型文件路径
        self.model_path = None

    # 加载整形数据，label为1到10，数据更新到self.data_tuple
    def load_data(self, test_size: float=0.33, random_state: int=1):
        if self.data is None:
            print('数据开始加载')
            self.data = datasets.load_digits()
            print('数据加载完成')
        assert (self.data is not None), 'self.data为None，请尝试传入可用data，或者执行load_digit_data'
        # 数据拆分训练集和测试集
        x, y = self.data['data'], self.data['target']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        self.data_tuple = x_train, x_test, y_train, y_test

    # 训练模型
    def train(self, param: dict=None, num_round: int=10):
        # 判定数据是否加载完成
        assert (self.data_tuple is not None), '数据尚未加载，无法训练，请先执行load_digit_data'
        print('模型开始预测')
        # 创建数据集
        x_train, x_test, y_train, y_test = self.data_tuple
        data_set = lgb.Dataset(x_train, label=y_train)
        if param is None:
            # 如果没有传参，默认10分类，类型为多分类
            param = {'num_class': 10, 'objective': 'multiclass'}
        # 训练轮数
        num_round = num_round
        print(f'模型传入的参数为：{param}，模型训练的轮数为：{num_round}')
        model = lgb.train(param, data_set, num_round)
        self.model = model
        print('模型训练完成')

    # 保存模型到文件中
    def save_model(self, model_path: str):
        # 校验模型是否训练过
        assert (self.model is not None), '模型尚未加载，或者模型类型不正确，请尝试先执行train或load_model_from_file，加载模型'
        # 保存模型
        self.model.save_model(filename=model_path)
        # 模型文件路径更新
        self.model_path = model_path

    # 从文件中加载模型
    def load_model_from_file(self, file_path: str):
        print(f'文件从模型中加载，加载路径为：{file_path}')
        model = lgb.Booster(model_file=file_path)
        self.model = model
        self.model_path = file_path
        print('模型加载完成')

    # 预测
    def predict(self):
        # 判定数据是否加载完成
        assert (self.data_tuple is not None), '数据尚未加载，无法预测，请先执行load_digit_data'
        assert (self.model is not None), '模型尚未加载，或者模型类型不正确，请尝试先执行train或load_model_from_file，加载模型'
        print('模型开始预测')
        # 预测
        x_train, x_test, y_train, y_test = self.data_tuple
        predict_data = self.model.predict(x_test)
        # 总数
        total_cnt = 0
        # 正确数量
        true_cnt = 0
        for i in range(len(y_test)):
            predict_cls = np.argmax(predict_data[i])
            label = y_test[i]
            softmax_rate = predict_data[i][predict_cls]
            is_true = bool(predict_cls == label)
            total_cnt += 1
            if is_true:
                true_cnt += 1
            print(f'预测分类为：{predict_cls}，实际分类为：{label}，预测softmax概率为：{round(softmax_rate, 2)}，分类结果：{is_true}')
        true_rate = round(true_cnt / total_cnt * 100, 2)
        print(f'样本总数：{total_cnt}，预测正确个数：{true_cnt}，正确率：{true_rate}%')
        print('预测完成\n')
        return predict_data


if __name__ == "__main__":
    # 实例化模型，可以传入data，如果不传入会加载sklearn.datasets.load_digits()数据
    ld = LgbDigits()
    # 加载数据，并将数据切分为训练集和测试集，可传参数：test_size=0.33（测试集所占比例）, random_state=1（随机种子）
    ld.load_data()
    # 训练模型，params={'num_class': 10, 'objective': 'multiclass'}（lightgbm训练参数），num_round=10（训练轮数）
    ld.train()
    # 预测数据
    ld.predict()
