import numpy as np
import pickle
from operator import mul
from functools import reduce


def bootstrap_training_set(x, y):
    input_num = x.shape[0]
    index = np.arange(input_num)
    bootstrapped_index = np.random.choice(index, input_num, replace=True)
    x_boot, y_boot = np.asarray([x[bootstrapped_index[i]] for i in range(0, len(bootstrapped_index))]) \
        , np.asarray([y[bootstrapped_index[i]] for i in range(0, len(bootstrapped_index))])
    return x_boot, y_boot, bootstrapped_index


if __name__ == '__main__':
    with open('//home//zichuliu//Desktop//hw1//homework-master//hw1//expert_data//HalfCheetah-v2.pkl', 'rb') as f:
        input = pickle.load(f)
    binput, btarget, index = bootstrap_training_set(input['observations'], input['actions'])
    test_case = [binput[i] == input['observations'][index[i]] for i in range(0, len(index))]
    print(reduce(mul, test_case))
    print(binput.shape)
