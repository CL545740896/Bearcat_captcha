import re


def project(string: str, work_path: str, project_name: str):
    string = re.sub('{', '{{', string)
    string = re.sub('}', '}}', string)
    string = re.sub(work_path, '{work_path}', string)
    string = re.sub(project_name, '{project_name}', string)
    print(string)


if __name__ == '__main__':
    string = """import os
import datetime

USE_GPU = True

MODE = 'CTC'

# 定义模型的方法,模型在models.py定义
MODEL = 'captcha_model_ctc'

# 学习率
LR = 1e-3

# 训练次数
EPOCHS = 100

# batsh批次
BATCH_SIZE = 32

# 训练多少轮验证损失下不去，学习率/10
LR_PATIENCE = 16

# 训练多少轮验证损失下不去，停止训练
EARLY_PATIENCE = 64

# 图片高度
IMAGE_HEIGHT = 40

# 图片宽度
IMAGE_WIDTH = 100

# 图片通道
IMAGE_CHANNALS = 3

# 验证码的长度
CAPTCHA_LENGTH = 6

# 保存的模型名称
MODEL_NAME = 'captcha.h5'

# 是否使用数据增强(数据集多的时候不需要用)
DATA_ENHANCEMENT = False

# 可视化配置batch或epoch
UPDATE_FREQ = 'epoch'

# 训练集路径
train_path = os.path.join(os.getcwd(), 'train_dataset')

# 增强后的路径
train_enhance_path = os.path.join(os.getcwd(), 'train_enhance_dataset')

# 验证集路径
validation_path = os.path.join(os.getcwd(), 'validation_dataset')

# 测试集路径
test_path = os.path.join(os.getcwd(), 'test_dataset')

# 标签路径
label_path = os.path.join(os.getcwd(), 'label')

# 打包训练集路径
TFRecord_train_path = os.path.join(os.getcwd(), 'train_pack_dataset')

# 打包验证集
TFRecord_validation_path = os.path.join(os.getcwd(), 'validation_pack_dataset')

# 打包测试集路径
TFRecord_test_path = os.path.join(os.getcwd(), 'test_pack_dataset')

# 模型保存路径
model_path = os.path.join(os.getcwd(), 'model')

# 可视化日志路径
log_dir = os.path.join(os.path.join(os.getcwd(), 'logs'), f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

# csv_logger日志路径
csv_path = os.path.join(os.path.join(os.getcwd(), 'CSVLogger'), 'traing.csv')

# 断点续训路径
checkpoint_path = os.path.join(os.getcwd(), 'checkpoint')  # 检查点路径

# checkpoint_file_path = os.path.join(checkpoint_path,
#                                     'Model_weights.-{epoch:02d}-{val_loss:.4f}.hdf5')
if MODE == 'CTC':
    checkpoint_file_path = os.path.join(checkpoint_path,
                                    'Model_weights.-{epoch:02d}-{val_loss:.4f}-{val_word_acc:.4f}.hdf5')
else:
    checkpoint_file_path = os.path.join(checkpoint_path,
                                    'Model_weights.-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5')
# TF训练集(打包后)
train_pack_path = os.path.join(os.getcwd(), 'train_pack_dataset')

# TF验证集(打包后)
validation_pack_path = os.path.join(os.getcwd(), 'validation_pack_dataset')

# TF测试集(打包后)
test_pack_path = os.path.join(os.getcwd(), 'test_pack_dataset')

# 提供后端放置的模型路径
App_model_path = os.path.join(os.getcwd(), 'App_model')

# 映射表
n_class_file = os.path.join(os.getcwd(), 'num_classes.json')

"""

    project(string, work_path='works', project_name='sougou')
