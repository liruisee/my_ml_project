# 检验参数缺失或多余
def check_kwargs(check_dict: dict, all_keys: set=None, need_keys: set=None):
    if all_keys is not None:
        extra_keys = check_dict.keys() - all_keys
        assert(len(extra_keys) == 0), f'传入了无需传入的参数：{extra_keys}，应传入的参数全集为：{all_keys}'
    if need_keys is not None:
        loss_keys = need_keys - check_dict.keys()
        assert (len(loss_keys) == 0), f'缺少必须传入的参数：{loss_keys}，必须传入的参数全集为：{need_keys}'
