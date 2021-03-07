from util import GeneralUtils
from config import Config
from model.CNN import CNN

if __name__=="__main__":
    config = Config(
        load_dic = True
    )
    utils = GeneralUtils(config)
    model = CNN(utils)
    model.BuildModel()
