import re
import os
import tensorflow as tf
from loguru import logger
from works.weibo_sougou.settings import log_dir
from works.weibo_sougou.settings import csv_path
from works.weibo_sougou.settings import UPDATE_FREQ
from works.weibo_sougou.settings import LR_PATIENCE
from works.weibo_sougou.settings import EARLY_PATIENCE
from works.weibo_sougou.settings import checkpoint_path
from works.weibo_sougou.settings import checkpoint_file_path
from tqdm.keras import TqdmCallback
from works.weibo_sougou.Function_API import Image_Processing

# 开启可视化的命令
'''
tensorboard --logdir "logs"
'''


# 回调函数官方文档
# https://keras.io/zh/callbacks/
class CallBack(object):
    @classmethod
    def calculate_the_best_weight(self):
        if os.listdir(checkpoint_path):
            value = Image_Processing.extraction_image(checkpoint_path)
            extract_num = [os.path.splitext(os.path.split(i)[-1])[0] for i in value]
            num = [re.split('-', i) for i in extract_num]
            accs = [float(i[-1]) for i in num]
            losses = [float('-' + str(abs(float(i[-2])))) for i in num]
            index = [acc + loss for acc, loss in zip(accs, losses)]
            model_dict = dict((ind, val) for ind, val in zip(index, value))
            return model_dict.get(max(index))
        else:
            logger.debug('没有可用的检查点')

    @classmethod
    def callback(self, model):
        call = []
        if os.path.exists(checkpoint_path):
            if os.listdir(checkpoint_path):
                logger.debug('load the model')
                model.load_weights(os.path.join(checkpoint_path, self.calculate_the_best_weight()))
                logger.debug(f'读取的权重为{os.path.join(checkpoint_path, self.calculate_the_best_weight())}')

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file_path,
                                                         verbose=1,
                                                         save_weights_only=True,
                                                         save_best_only=True, period=1)
        call.append(cp_callback)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True,
                                                              update_freq=UPDATE_FREQ)
        call.append(tensorboard_callback)

        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(factor=0.01, patience=LR_PATIENCE)
        call.append(lr_callback)

        csv_callback = tf.keras.callbacks.CSVLogger(filename=csv_path, append=True)
        call.append(csv_callback)

        early_callback = tf.keras.callbacks.EarlyStopping(min_delta=0, verbose=1, patience=EARLY_PATIENCE)
        call.append(early_callback)
        call.append(TqdmCallback())
        return (model, call)


if __name__ == '__main__':
    logger.debug(CallBack.calculate_the_best_weight())
