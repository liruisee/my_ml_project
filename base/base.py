from abc import ABC, abstractmethod


# 基础执行器
class BaseExecutor(ABC):

    @abstractmethod
    def load_data(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        raise NotImplementedError()
