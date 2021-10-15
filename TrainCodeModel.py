import sys
import pickle
module_path = "$MODULE_PATH"
sys.path.append(module_path)

if __name__ == '__main__':
    from packages.yolov5 import data_train
    option = pickle.load(open("option", 'rb'))
    data_train.start_train(option)
