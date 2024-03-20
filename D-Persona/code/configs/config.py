# configuration for the models
import yaml


class Config:
    def __init__(self, config_path):

        with open(config_path, encoding='utf-8') as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

        # ----------- parse yaml ---------------#
        self.DATA_PATH = yaml_dict['DATA_PATH']
        self.DATASET = yaml_dict['DATASET']

        self.INPUT_CHANNEL = yaml_dict['INPUT_CHANNEL']
        self.OUTPUT_CHANNEL = yaml_dict['OUTPUT_CHANNEL']

        self.INPUT_SIZE = 128
        self.PATCH_SIZE = [128, 128]
        self.KFOLD = 4

        self.RANDOM_SEED = yaml_dict['RANDOM_SEED']
        self.MODEL_DIR = yaml_dict['MODEL_DIR']

        self.TRAIN_BATCHSIZE = yaml_dict['TRAIN_BATCHSIZE']
        self.VAL_BATCHSIZE = yaml_dict['VAL_BATCHSIZE']

        self.PRT_LOSS = yaml_dict['PRT_LOSS']
        self.VISUALIZE = yaml_dict['VISUALIZE']
        self.TEST_SAVE = yaml_dict['TEST_SAVE']

if __name__ == '__main__':
    cfg = Config(config_path='./params.yaml')
    print(cfg)