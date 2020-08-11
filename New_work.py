import os


def callback(work_path, project_name):
    return f"""import re
import os
import tensorflow as tf
from loguru import logger
from {work_path}.{project_name}.settings import log_dir
from {work_path}.{project_name}.settings import csv_path
from {work_path}.{project_name}.settings import UPDATE_FREQ
from {work_path}.{project_name}.settings import LR_PATIENCE
from {work_path}.{project_name}.settings import EARLY_PATIENCE
from {work_path}.{project_name}.settings import checkpoint_path
from {work_path}.{project_name}.settings import checkpoint_file_path
from tqdm.keras import TqdmCallback
from {work_path}.{project_name}.Function_API import Image_Processing

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
                logger.debug(f'读取的权重为{{os.path.join(checkpoint_path, self.calculate_the_best_weight())}}')

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
"""


def app(work_path, project_name):
    return f"""import os
import json
import operator
import tensorflow as tf
from flask import Flask
from flask import request
from loguru import logger
from {work_path}.{project_name}.models import Models
from {work_path}.{project_name}.Callback import CallBack
from {work_path}.{project_name}.settings import MODEL
from {work_path}.{project_name}.settings import MODEL_NAME
from {work_path}.{project_name}.settings import checkpoint_path
from {work_path}.{project_name}.settings import App_model_path
from {work_path}.{project_name}.settings import MULITI_MODEL_PREDICTION
from {work_path}.{project_name}.Function_API import Distinguish_image
from {work_path}.{project_name}.Function_API import Image_Processing

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

app = Flask(__name__)
if os.listdir(App_model_path):
    model_path = Image_Processing.extraction_image(App_model_path)
    logger.debug(f'{{model_path}}模型加载成功')
else:
    model = operator.methodcaller(MODEL)(Models)
    try:
        model.load_weights(os.path.join(checkpoint_path, CallBack.calculate_the_best_weight()))
    except:
        raise OSError(f'没有任何的权重和模型在{{App_model_path}}')
    model_save = os.path.join(App_model_path, MODEL_NAME)
    model.save(model_save)
    model_path = [model_save]
    logger.debug(f'{{model_path}}模型加载成功')


# logger.debug(model_path)

@app.route("/", methods=['POST'])
def captcha_predict():
    return_dict = {{'return_code': '200', 'return_info': '处理成功', 'result': False, 'recognition_rate': 0,
                   'overall_recognition_rate': 0}}
    get_data = request.form.to_dict()
    if 'img' in get_data.keys():
        base64_str = request.form['img']
        try:
            if MULITI_MODEL_PREDICTION:
                overall_recognition_rate, recognition_rate, lable_forecast = Distinguish_image.distinguish_apis(
                    model_path=model_path,
                    base64_str=base64_str)
                return_dict['result'] = lable_forecast
                return_dict['recognition_rate'] = recognition_rate
                return_dict['overall_recognition_rate'] = overall_recognition_rate
            else:
                overall_recognition_rate, recognition_rate, lable_forecast = Distinguish_image.distinguish_api(
                    model_path=model_path,
                    base64_str=base64_str)
                return_dict['result'] = lable_forecast
                return_dict['recognition_rate'] = recognition_rate
                return_dict['overall_recognition_rate'] = overall_recognition_rate
        except Exception as e:
            return_dict['result'] = str(e)
            return_dict['return_info'] = '模型识别错误'
    else:
        return_dict['return_code'] = '5004'
        return_dict['return_info'] = '参数错误，没有img属性'
    return json.dumps(return_dict, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)
"""


def captcha_config():
    return '''{
  "train_dir": "train_dataset",
  "validation_dir": "validation_dataset",
  "test_dir": "test_dataset",
  "image_suffix": "jpg",
  "characters": "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
  "count": 20000,
  "char_count": [4,5,6],
  "width": 100,
  "height": 60
}
'''


def del_file(work_path, project_name):
    return f'''# 增强后文件太多，手动删非常困难，直接用代码删
import shutil
from tqdm import tqdm
from loguru import logger
from {work_path}.{project_name}.settings import train_path
from {work_path}.{project_name}.settings import test_path
from {work_path}.{project_name}.settings import validation_path
from {work_path}.{project_name}.settings import train_enhance_path
from {work_path}.{project_name}.settings import train_pack_path
from {work_path}.{project_name}.settings import validation_pack_path
from {work_path}.{project_name}.settings import test_pack_path
from concurrent.futures import ThreadPoolExecutor


def del_file(path):
    try:
        shutil.rmtree(path)
        logger.debug(f'成功删除{{path}}')
    except WindowsError as e:
        logger.error(e)


if __name__ == '__main__':
    path = [train_path, test_path, validation_path, train_enhance_path, train_pack_path, validation_pack_path,
            test_pack_path]
    with ThreadPoolExecutor(max_workers=7) as t:
        for i in tqdm(path, desc='正在删除'):
            t.submit(del_file, i)
'''


def Function_API(work_path, project_name):
    return f"""# 函数丢这里
import re
import os
import time
import json
import glob
import shutil
import base64
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from loguru import logger
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from {work_path}.{project_name}.settings import MODE
from {work_path}.{project_name}.settings import validation_path
from {work_path}.{project_name}.settings import test_path
from {work_path}.{project_name}.settings import N_CLASS
from {work_path}.{project_name}.settings import IMAGE_HEIGHT
from {work_path}.{project_name}.settings import IMAGE_WIDTH
from {work_path}.{project_name}.settings import CAPTCHA_LENGTH
from {work_path}.{project_name}.settings import IMAGE_CHANNALS
from {work_path}.{project_name}.settings import CAPTCHA_CHARACTERS_LENGTH
from concurrent.futures import ThreadPoolExecutor


class Image_Processing(object):
    # 分割数据集
    @classmethod
    def move_path(self, path: list, proportion=0.2) -> bool:
        logger.debug(f'数据集有{{len(path)}},{{proportion * 100}}%作为验证集,{{proportion * 100}}%作为测试集')
        division_number = int(len(path) * proportion)
        logger.debug(f'验证集数量为{{division_number}},测试集数量为{{division_number}}')
        validation_dataset = random.sample(path, division_number)
        for i in tqdm(validation_dataset, desc='准备移动'):
            path.remove(i)
        validation = [os.path.join(validation_path, os.path.split(i)[-1]) for i in validation_dataset]
        logger.debug(validation)
        with ThreadPoolExecutor(max_workers=5) as t:
            for full_path, des_path in zip(validation_dataset, validation):
                t.submit(shutil.move, full_path, des_path)
        test_dataset = random.sample(path, division_number)
        test = [os.path.join(test_path, os.path.split(i)[-1]) for i in test_dataset]
        with ThreadPoolExecutor(max_workers=5) as t:
            for full_path, des_path in zip(test_dataset, test):
                t.submit(shutil.move, full_path, des_path)
        logger.info(f'任务结束')
        return True

    # 修改文件名
    @classmethod
    def rename_path(self, path: list, original='.', reform='_'):
        for i in tqdm(path, desc='正在改名'):
            paths, name = os.path.split(i)
            name, mix = os.path.splitext(name)
            if original in name:
                new_name = name.replace(original, reform)
                os.rename(i, os.path.join(paths, new_name + mix))

    @classmethod
    def rename_suffix(self, path: list):
        for i in tqdm(path, desc='正在修改后缀'):
            paths, name = os.path.split(i)
            name, mix = os.path.splitext(name)
            os.rename(i, os.path.join(paths, name + '.jpg'))

    @classmethod
    # 提取全部图片plus
    def extraction_image(self, path: str, mode=MODE, shuffix='jpg') -> list:
        if mode == 'tagging':
            data_path = glob.glob(f'{{path}}/*.{{shuffix}}')
            lable_file = glob.glob(f'{{path}}/*.xlsx')[0]
            df = pd.read_excel(lable_file, header=None)
            df_path = [os.path.split(i)[-1] for i in df[0]]
            df_lable = df[1]
            dicts = dict((index, name) for index, name in zip(df_path, df_lable))
            with open('n_class.json', 'w', encoding='utf-8') as f:
                f.write(json.dumps(dicts, ensure_ascii=False))
            return data_path
        else:
            try:
                data_path = []
                datas = [os.path.join(path, i) for i in os.listdir(path)]
                for data in datas:
                    data_path = data_path + [os.path.join(data, i) for i in os.listdir(data)]
                return data_path
            except:
                return [os.path.join(path, i) for i in os.listdir(path)]

    # 增强图片
    @classmethod
    def preprosess_save_images(self, image_path, size):
        logger.debug(f'开始处理{{image_path}}')
        image_name, image_suffix = os.path.splitext(os.path.split(image_path)[-1])
        img_raw = tf.io.read_file(image_path)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNALS)
        # img_tensor_up = tf.image.flip_up_down(img_tensor)
        # img_tensor_a = tf.image.resize(img_tensor, size)
        # 旋转
        # img_tensor_rotated_90 = tf.image.resize(tf.image.rot90(img_tensor), size)
        # img_tensor_rotated_180 = tf.image.resize(tf.image.rot90(tf.image.rot90(img_tensor)), size)
        # img_tensor_rotated_270 = tf.image.resize(tf.image.rot90(tf.image.rot90(tf.image.rot90(img_tensor))), size)
        # 对比度
        img_tensor_contrast1 = tf.image.resize(tf.image.adjust_contrast(img_tensor, 1), size)

        img_tensor_contrast9 = tf.image.resize(tf.image.adjust_contrast(img_tensor, 9), size)
        # 饱和度
        img_tensor_saturated_1 = tf.image.resize(tf.image.adjust_saturation(img_tensor, 1), size)

        img_tensor_saturated_9 = tf.image.resize(tf.image.adjust_saturation(img_tensor, 9), size)
        # 亮度
        img_tensor_brightness_1 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.1), size)

        img_tensor_brightness_4 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.4), size)
        # img_tensor_brightness_5 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.5), size)
        # img_tensor_brightness_6 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.6), size)
        # img_tensor_brightness_7 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.7), size)
        # img_tensor_brightness_8 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.8), size)
        # img_tensor_brightness_9 = tf.image.resize(tf.image.adjust_brightness(img_tensor, 0.9), size)
        # 裁剪
        # img_tensor_crop1 = tf.image.resize(tf.image.central_crop(img_tensor, 0.1), size)
        # img_tensor_crop2 = tf.image.resize(tf.image.central_crop(img_tensor, 0.2), size)
        # img_tensor_crop3 = tf.image.resize(tf.image.central_crop(img_tensor, 0.3), size)
        # img_tensor_crop4 = tf.image.resize(tf.image.central_crop(img_tensor, 0.4), size)
        # img_tensor_crop5 = tf.image.resize(tf.image.central_crop(img_tensor, 0.5), size)
        # 调整色相
        img_tensor_hue1 = tf.image.resize(tf.image.adjust_hue(img_tensor, 0.1), size)

        img_tensor_hue9 = tf.image.resize(tf.image.adjust_hue(img_tensor, 0.9), size)
        # 图片标准化
        img_tensor_standardization = tf.image.resize(tf.image.per_image_standardization(img_tensor), size)
        # img_tensor = tf.cast(img_tensor, tf.float32)
        # img_tensor = img_tensor / 255
        image_tensor = [img_tensor_contrast1, img_tensor_contrast9, img_tensor_saturated_1, img_tensor_saturated_9,
                        img_tensor_brightness_1, img_tensor_brightness_4, img_tensor_hue1, img_tensor_hue9,
                        img_tensor_standardization]
        for index, i in tqdm(enumerate(image_tensor), desc='正在生成图片'):
            img_tensor = np.asarray(i.numpy(), dtype='uint8')
            img_tensor = tf.image.encode_jpeg(img_tensor)
            with tf.io.gfile.GFile(f'train_enhance_dataset/{{image_name}}_{{str(index)}}{{image_suffix}}', 'wb') as file:
                file.write(img_tensor.numpy())
        logger.info(f'处理完成{{image_path}}')
        return True

    @classmethod
    # 展示图片处理后的效果
    def show_image(self, image_path, mode=MODE):
        '''
        展示图片处理后的效果
        :param image_path:
        :return:
        '''
        if mode == 'tagging':
            im = tf.io.read_file(image_path)
            img_tensor = tf.image.decode_image(im)
            img_tensor = tf.cast(img_tensor, tf.float32)
            img_tensor = np.asarray(img_tensor.numpy(), dtype='uint8')
            # print(img_tensor.shape)
            # print(img_tensor.dtype)
            plt.imshow(img_tensor)
            # plt.show()
            xmax, ymax, xmin, ymin = [183, 33, 68, 70]
            rect = Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), fill=False, color='red')
            ax = plt.gca()
            ax.axes.add_patch(rect)
            plt.show()
        else:
            img_raw = tf.io.read_file(image_path)
            img_tensor = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNALS)
            img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
            img_tensor = tf.cast(img_tensor, tf.float32)
            img_tensor = np.asarray(img_tensor.numpy(), dtype='uint8')
            print(img_tensor.shape)
            print(img_tensor.dtype)
            plt.imshow(img_tensor)
            plt.show()

    @classmethod
    # 对图片进行解码,预测
    def load_image(self, path):
        '''
        预处理图片函数
        :param path:图片路径
        :return: 处理好的路径
        '''
        img_raw = tf.io.read_file(path)
        # channel=3 是彩色图片
        img_tensor = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor = img_tensor / 255.
        img_tensor = tf.expand_dims(img_tensor, 0)
        return img_tensor

    @classmethod
    def char2pos(self, c):
        c = str(c)
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    # 向量转文本
    @classmethod
    def vector2text(self, vector):
        char_pos = vector.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            char_idx = c % 63
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            text.append(chr(char_code))
        return "".join(text)

    @classmethod
    def lable2vector_ocr(self, text):
        vector = np.zeros(CAPTCHA_LENGTH * N_CLASS)
        for i, c in enumerate(text):
            index = i * N_CLASS + c
            vector[index] = 1
        return vector

    # 文本转向量
    @classmethod
    def text2vector(self, text, mode='ordinary', dicts=None):
        if mode == 'ordinary':
            if len(text) < CAPTCHA_LENGTH:
                while True:
                    text = text + '_'
                    if len(text) == CAPTCHA_LENGTH:
                        break
                    else:
                        continue
            if len(text) > CAPTCHA_LENGTH:
                raise ValueError(f'有验证码长度大于{{CAPTCHA_LENGTH}}标签为:{{text}}')
            # 10个数字，大小写字母26,一个_表示不足CAPTCHA_LENGTH
            vector = np.zeros(CAPTCHA_LENGTH * CAPTCHA_CHARACTERS_LENGTH)
            for i, c in enumerate(text):
                index = i * CAPTCHA_CHARACTERS_LENGTH + self.char2pos(c)
                vector[index] = 1
            return vector
        elif mode == 'n_class':
            ver = np.zeros(N_CLASS)
            ver[text] = 1
            return ver
        elif mode == 'ordinary_ocr':
            string = str(text)
            vector_ocr = []
            while True:
                if len(text) < CAPTCHA_LENGTH:
                    string = string + '_'
                    continue
                if len(string) == CAPTCHA_LENGTH:
                    break
                if len(string) > CAPTCHA_LENGTH:
                    raise ValueError(f'字符长度{{len(string)}}大于设置{{CAPTCHA_LENGTH}}')
            for i in string:
                lable = dicts.get(i)
                vector_ocr.append(lable)
            # vector_ocr = np.array(vector_ocr)
            return vector_ocr
        else:
            raise ValueError(f'没有mode={{mode}}提取标签的方法')

    @classmethod
    def extraction_lable(self, path_list: list, suffix=True, divide='_', mode=MODE):
        if mode == 'ordinary':
            if suffix:
                lable_list = [re.split(divide, os.path.splitext(os.path.split(i)[-1])[0])[0] for i in
                              tqdm(path_list, desc='正在获取文件名')]
                lable_list = [self.text2vector(i, mode=mode) for i in tqdm(lable_list, desc='正在生成numpy')]
                return lable_list
            else:
                lable_list = [os.path.splitext(os.path.split(i)[-1])[0] for i in tqdm(path_list, desc='正在获取文件名')]
                lable_list = [self.text2vector(i, mode=mode) for i in tqdm(lable_list, desc='正在生成numpy')]
                return lable_list
        elif mode == 'n_class':
            if suffix:
                # dicts = self.extraction_dict(path_list)
                # logger.debug(f'一共有{{len(dicts)}}类')
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
                path = [re.split(divide, i)[0] for i in paths]
                dicts_list = sorted(set(path))
                logger.debug(f'一共有{{len(dicts_list)}}类')
                # paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
                # path = [re.split(divide, i)[0] for i in paths]
                # dicts = sorted(set(path))
                dicts = dict((name, index) for index, name in enumerate(dicts_list))
                d = dict((index, name) for index, name in enumerate(dicts_list))
                with open('n_class.json', 'w', encoding='utf-8') as f:
                    f.write(json.dumps(d, ensure_ascii=False))
                lable_list = [self.text2vector(dicts.get(i), mode=mode) for i in path]
                return lable_list
            else:
                # dicts = self.extraction_dict(path_list)
                # logger.debug(f'一共有{{len(dicts)}}类')
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
                dicts_list = sorted(set(paths))
                # dicts_list = sorted(set(path))
                logger.debug(f'一共有{{len(dicts_list)}}类')
                dicts = dict((name, index) for index, name in enumerate(dicts_list))
                d = dict((index, name) for index, name in enumerate(dicts_list))
                with open('n_class.json', 'w', encoding='utf-8') as f:
                    f.write(json.dumps(d, ensure_ascii=False))
                lable_list = [self.text2vector(dicts.get(i), mode=mode) for i in paths]
                return lable_list
        elif mode == 'ordinary_ocr':
            if suffix:
                # dicts = self.extraction_ocr_dict(path_list, divide=divide)
                # logger.debug(f'一共有{{len(dicts)}}类')
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
                path = [re.split(divide, i)[0] for i in paths]
                ocr_path = []
                for i in path:
                    for s in i:
                        ocr_path.append(s)
                # ocr_path = self.text2index(path)
                dicts_list = sorted(set(ocr_path))
                d = dict((index, name) for index, name in enumerate(dicts_list))
                with open('n_class.json', 'w', encoding='utf-8') as f:
                    f.write(json.dumps(d, ensure_ascii=False))
                dicts = dict((name, index) for index, name in enumerate(dicts_list))
                dicts['_'] = len(dicts)

                lable_list = [self.text2vector(i, mode=mode, dicts=dicts) for i in path]
                lable_list = [self.lable2vector_ocr(i) for i in lable_list]
                return lable_list
            else:
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
                ocr_path = []
                for i in paths:
                    for s in i:
                        ocr_path.append(s)
                dicts_list = sorted(set(ocr_path))
                d = dict((index, name) for index, name in enumerate(dicts_list))
                with open('n_class.json', 'w', encoding='utf-8') as f:
                    f.write(json.dumps(d, ensure_ascii=False))
                dicts = dict((name, index) for index, name in enumerate(dicts_list))
                dicts['_'] = len(dicts)
                lable_list = [self.text2vector(i, mode=mode, dicts=dicts) for i in paths]
                lable_list = [self.lable2vector_ocr(i) for i in lable_list]
                return lable_list
        elif mode == 'tagging':
            with open('n_class.json', 'r') as f:
                dicts = json.loads(f.read())
            lable_list = [(os.path.split(name)[-1], Image.open(size).size) for name, size in zip(path_list, path_list)]
            logger.debug(lable_list)
        else:
            raise ValueError(f'没有mode={{mode}}提取标签的方法')


# 打包数据
class WriteTFRecord(object):
    @classmethod
    def WriteTFRecord(self, TFRecord_path, datasets: list, lables: list, file_name='dataset.tfrecords', mode=MODE):
        num_count = len(datasets)
        lables_count = len(lables)
        if not os.path.exists(TFRecord_path):
            os.mkdir(TFRecord_path)
        logger.info(f'文件个数为:{{num_count}}')
        logger.info(f'标签个数为:{{lables_count}}')
        filename = os.path.join(TFRecord_path, file_name)
        writer = tf.io.TFRecordWriter(filename)
        logger.info(f'开始保存{{filename}}')
        for dataset, lable in zip(datasets, lables):
            image_bytes = open(dataset, 'rb').read()
            num_count = num_count - 1
            logger.debug(f'剩余{{num_count}}图片待打包')
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={{'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                             'lable': tf.train.Feature(float_list=tf.train.FloatList(value=lable))}}))
            # 序列化
            serialized = example.SerializeToString()
            writer.write(serialized)
        logger.info(f'保存{{filename}}成功')
        writer.close()
        return filename


@tf.function
# 处理图片(将图片转化成tensorflow)
def load_preprosess_image(image_path):
    '''
    处理图片
    :param image_path:一张图片的路径
    :return:tensor
    '''
    img_raw = tf.io.read_file(image_path)
    img_tensor = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNALS)
    img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = img_tensor / 255.
    return img_tensor


@tf.function
# 映射函数
def parse_function_n_class(exam_proto):
    features = {{
        'image': tf.io.FixedLenFeature([], tf.string),
        'lable': tf.io.FixedLenFeature((80,), tf.float32)
    }}
    parsed_example = tf.io.parse_single_example(exam_proto, features)
    img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
    img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
    img_tensor = img_tensor / 255.
    lable_tensor = parsed_example['lable']
    return (img_tensor, lable_tensor)


@tf.function
# 映射函数
def parse_function(exam_proto, mode=MODE):
    if mode == 'ordinary':
        features = {{
            'image': tf.io.FixedLenFeature([], tf.string),
            'lable': tf.io.FixedLenFeature([CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH], tf.float32)
        }}
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        lable_tensor = parsed_example['lable']
        return (img_tensor, lable_tensor)
    elif mode == 'n_class':
        features = {{
            'image': tf.io.FixedLenFeature([], tf.string),
            'lable': tf.io.FixedLenFeature([N_CLASS], tf.float32)
        }}
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        lable_tensor = parsed_example['lable']
        return (img_tensor, lable_tensor)
    elif mode == 'ordinary_ocr':
        features = {{
            'image': tf.io.FixedLenFeature([], tf.string),
            'lable': tf.io.FixedLenFeature([CAPTCHA_LENGTH, N_CLASS], tf.float32)
        }}
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        lable_tensor = parsed_example['lable']
        return (img_tensor, lable_tensor)
    else:
        raise ValueError(f'没有mode={{mode}}映射的方法')


class Distinguish_image(object):
    true_value = 0
    predicted_value = 0

    @classmethod
    def probability(self, vector):
        return max(vector) / reduce(lambda x, y: x + y, vector)

    @classmethod
    def char2pos(self, c):
        c = str(c)
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    @classmethod
    def text2vector(self, text):
        if len(text) < CAPTCHA_LENGTH:
            while True:
                text = text + '_'
                if len(text) == CAPTCHA_LENGTH:
                    break
                else:
                    continue
        if len(text) > CAPTCHA_LENGTH:
            raise ValueError(f'有验证码长度大于{{CAPTCHA_LENGTH}}标签为:{{text}}')
        # 10个数字，大小写字母26,一个_表示不足CAPTCHA_LENGTH
        vector = np.zeros(CAPTCHA_LENGTH * CAPTCHA_CHARACTERS_LENGTH)
        for i, c in enumerate(text):
            index = i * CAPTCHA_CHARACTERS_LENGTH + self.char2pos(c)
            vector[index] = 1
        return vector

    @classmethod
    def vector2text(self, vector):
        char_pos = tf.argmax(vector, axis=1)
        overall_recognition_rate = []
        recognition_rate = []
        text = []
        for v, c in zip(vector, char_pos):
            char_idx = c % 63
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            recognition = self.probability(v)
            if chr(char_code) != '_':
                recognition_rate.append({{chr(char_code): '%.2f' % (recognition * 100) + '%'}})
                overall_recognition_rate.append(recognition)
            text.append(chr(char_code))
        return ('%.2f' % (min(overall_recognition_rate) * 100) + '%', recognition_rate, "".join(text))

    @classmethod
    def extraction_lable_name(self, path, suffix=True, divide='_'):
        if suffix:
            lable_list = re.split(divide, os.path.splitext(os.path.split(path)[-1])[0])[0]
            return lable_list
        else:
            lable_list = os.path.splitext(os.path.split(path)[-1])[0]
            return lable_list

    @classmethod
    def model_predict(self, model, jpg_path):
        return tf.keras.models.load_model(model).predict(Image_Processing.load_image(jpg_path))

    # 预测方法
    @classmethod
    def distinguish_images(self, model_path, jpg_path, suffix=True, divide='_'):
        vertor = []
        with ThreadPoolExecutor() as t:
            for model in model_path:
                result = t.submit(self.model_predict, model, jpg_path)
                vertor.append(result.result())
        vertor_sum = reduce(lambda x, y: x + y, vertor)
        forecast = vertor_sum[0] / len(vertor)
        overall_recognition_rate, recognition_rate, lable_forecast = self.vector2text(forecast)
        lable_real = self.extraction_lable_name(jpg_path, suffix, divide)
        logger.info(f'预测值为{{lable_forecast.replace("_", "")}},真实值为{{lable_real.replace("_", "")}}')
        logger.info(f'每个字符的识别率:{{recognition_rate}}')
        logger.info(f'整体识别率{{overall_recognition_rate}}')
        if str(lable_forecast.replace("_", "")) != str(lable_real.replace("_", "")):
            logger.error(f'预测失败的图片路径为:{{jpg_path}}')
            self.true_value = self.true_value + 1
            logger.debug(f'正确率:{{(self.predicted_value / self.true_value) * 100}}%')
            if self.predicted_value > 0:
                logger.debug(f'预测正确{{self.predicted_value}}张图片')
        else:
            self.predicted_value = self.predicted_value + 1
            self.true_value = self.true_value + 1
            logger.debug(f'正确率:{{(self.predicted_value / self.true_value) * 100}}%')
            if self.predicted_value > 0:
                logger.debug(f'预测正确{{self.predicted_value}}张图片')
        return lable_forecast

    # 预测方法
    @classmethod
    def distinguish_image(self, model_path, jpg_path, suffix=True, divide='_', mode=MODE):
        if mode == 'ordinary':
            model = tf.keras.models.load_model(model_path)
            forecast = model.predict(Image_Processing.load_image(jpg_path))
            overall_recognition_rate, recognition_rate, lable_forecast = self.vector2text(forecast[0])
            # logger.debug(recognition_rate)
            lable_real = self.extraction_lable_name(jpg_path, suffix, divide)
            logger.info(f'预测为{{lable_forecast.replace("_", "")}},真实为{{lable_real.replace("_", "")}}')
            logger.info(f'每个字符的识别率:{{recognition_rate}}')
            logger.info(f'整体识别率{{overall_recognition_rate}}')
            if str(lable_forecast.replace("_", "")) != str(lable_real.replace("_", "")):
                logger.error(f'预测失败的图片路径为:{{jpg_path}}')
                self.true_value = self.true_value + 1
                logger.debug(f'正确率:{{(self.predicted_value / self.true_value) * 100}}%')
                if self.predicted_value > 0:
                    logger.debug(f'预测正确{{self.predicted_value}}张图片')
            else:
                self.predicted_value = self.predicted_value + 1
                self.true_value = self.true_value + 1
                logger.debug(f'正确率:{{(self.predicted_value / self.true_value) * 100}}%')
                if self.predicted_value > 0:
                    logger.debug(f'预测正确{{self.predicted_value}}张图片')
            return lable_forecast
        elif mode == 'n_class':
            model = tf.keras.models.load_model(model_path)
            forecast = model.predict(Image_Processing.load_image(jpg_path))
            lable_real = self.extraction_lable_name(jpg_path, suffix, divide)
            recognition_rate, lable_forecast = Distinguish_image.vector2lable_name(forecast[0])
            logger.info(f'预测值为{{lable_forecast.replace("_", "")}},真实值为{{lable_real.replace("_", "")}}')
            logger.info(f'图片的识别率:{{recognition_rate}}')
            if str(lable_forecast.replace("_", "")) != str(lable_real.replace("_", "")):
                logger.error(f'预测失败的图片路径为:{{jpg_path}}')
                self.true_value = self.true_value + 1
                logger.debug(f'正确率:{{(self.predicted_value / self.true_value) * 100}}%')
                if self.predicted_value > 0:
                    logger.debug(f'预测正确{{self.predicted_value}}张图片')
            else:
                self.predicted_value = self.predicted_value + 1
                self.true_value = self.true_value + 1
                logger.debug(f'正确率:{{(self.predicted_value / self.true_value) * 100}}%')
                if self.predicted_value > 0:
                    logger.debug(f'预测正确{{self.predicted_value}}张图片')
            return lable_forecast
        elif mode == 'ordinary_ocr':
            pass
        else:
            raise ValueError(f'没有{{mode}}的预测方法')

    @classmethod
    def vector2lable_name(self, vector):
        char_pos = np.argmax(vector)
        with open('n_class.json', 'r', encoding='utf-8') as f:
            n_class_dict = json.loads(f.read())
        lable = n_class_dict.get(str(char_pos))
        recognition_rate = Distinguish_image.probability(vector)
        return (recognition_rate, lable)

    @classmethod
    # 对图片进行解码,预测
    def load_image(self, img_raw):
        '''
        预处理图片函数
        :param path:图片路径
        :return: 处理好的路径
        '''
        # img_raw = tf.io.decode_jpeg(path)
        # channel 是彩色图片
        img_tensor = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor = img_tensor / 255.
        img_tensor = tf.expand_dims(img_tensor, 0)
        return img_tensor

    # 后端
    @classmethod
    def distinguish_api(self, model_path, base64_str):
        '''
        with open(file_name,'rb') as f:
            base64_str = base64.b64encode(f.read()).decode('utf-8')
        :param model_path:
        :param base64_str:
        :return:
        '''
        jpg = base64.b64decode(base64_str)
        model = tf.keras.models.load_model(model_path[0])
        forecast = model.predict(Distinguish_image.load_image(jpg))
        overall_recognition_rate, recognition_rate, lable_forecast = self.vector2text(forecast[0])
        return (overall_recognition_rate, recognition_rate, lable_forecast.replace("_", ""))

    @classmethod
    def model_app(self, model, jpg):
        return tf.keras.models.load_model(model).predict(Distinguish_image.load_image(jpg))

    # 后端
    @classmethod
    def distinguish_apis(self, model_path: list, base64_str):
        '''
        with open(file_name,'rb') as f:
            base64_str = base64.b64encode(f.read()).decode('utf-8')
        :param model_path:
        :param base64_str:
        :return:
        '''
        jpg = base64.b64decode(base64_str)
        # model = tf.keras.models.load_model(model_path)
        # forecast = model.predict(Distinguish_image.load_image(jpg))
        vertor = []
        with ThreadPoolExecutor() as t:
            for model in model_path:
                result = t.submit(self.model_app, model, jpg)
                vertor.append(result.result())
        # vertor = [tf.keras.models.load_model(model).predict(Distinguish_image.load_image(jpg)) for model in
        #           model_path]
        vertor_sum = reduce(lambda x, y: x + y, vertor)
        forecast = vertor_sum[0] / len(vertor)
        overall_recognition_rate, recognition_rate, lable_forecast = self.vector2text(forecast)
        # lable_forecast = self.vector2text(forecast[0])
        return (overall_recognition_rate, recognition_rate, lable_forecast.replace("_", ""))


def cheak_path(path):
    while True:
        if os.path.exists(path):
            paths, name = os.path.split(path)
            name, mix = os.path.splitext(name)
            name = name + f'_{{int(time.time())}}'
            path = os.path.join(paths, name + mix)
        if not os.path.exists(path):
            return path

"""


def gen_sample_by_captcha(work_path, project_name):
    return '''# -*- coding: UTF-8 -*-
"""
使用captcha lib生成验证码（前提：pip install captcha）
"""

import os
import time
import json
import random
from tqdm import tqdm
from captcha.image import ImageCaptcha
from concurrent.futures import ThreadPoolExecutor


def gen_special_img(text, file_path, width, height):
    # 生成img文件
    generator = ImageCaptcha(width=width, height=height)  # 指定大小
    img = generator.generate_image(text)  # 生成图片
    img.save(file_path)  # 保存图片


def gen_ima_by_batch(root_dir, image_suffix, characters, count, char_count, width, height):
    # 判断文件夹是否存在
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for _ in tqdm(enumerate(range(count)), desc='Generate captcha image'):
        text = ""
        for _ in range(random.choice(char_count)):
            text += random.choice(characters)

        timec = str(time.time()).replace(".", "")
        p = os.path.join(root_dir, "{}_{}.{}".format(text, timec, image_suffix))
        gen_special_img(text, p, width, height)

        # logger.debug("Generate captcha image => {}".format(index + 1))


def main():
    with open("captcha_config.json", "r") as f:
        config = json.load(f)
    # 配置参数
    train_dir = config["train_dir"]
    validation_dir = config["validation_dir"]
    test_dir = config["test_dir"]
    image_suffix = config["image_suffix"]  # 图片储存后缀
    characters = config["characters"]  # 图片上显示的字符集 # characters = "0123456789abcdefghijklmnopqrstuvwxyz"
    count = config["count"]  # 生成多少张样本
    char_count = config["char_count"]  # 图片上的字符数量

    # 设置图片高度和宽度
    width = config["width"]
    height = config["height"]

    with ThreadPoolExecutor(max_workers=3) as t:
        t.submit(gen_ima_by_batch, train_dir, image_suffix, characters, count, char_count, width, height)
        t.submit(gen_ima_by_batch, validation_dir, image_suffix, characters, count, char_count, width, height)
        t.submit(gen_ima_by_batch, test_dir, image_suffix, characters, count, char_count, width, height)


if __name__ == '__main__':
    main()
'''


def init_working_space(work_path, project_name):
    return f'''# 检查项目路径
import os
import shutil
from tqdm import tqdm
from loguru import logger
from {work_path}.{project_name}.settings import App_model_path
from {work_path}.{project_name}.settings import checkpoint_path


def chrak_path():
    path = os.getcwd()
    paths = ['test_dataset', 'train_dataset', 'validation_dataset', 'train_enhance_dataset', 'train_pack_dataset',
             'validation_pack_dataset', 'test_pack_dataset', 'model', 'logs', 'CSVLogger', checkpoint_path,
             App_model_path]
    for i in tqdm(paths, desc='正在创建文件夹'):
        mix = os.path.join(path, i)
        if not os.path.exists(mix):
            os.mkdir(mix)


def del_file():
    path = [os.path.join(os.getcwd(), 'CSVLogger'),
            os.path.join(os.getcwd(), 'logs'), checkpoint_path]
    for i in tqdm(path, desc='正在删除'):
        try:
            shutil.rmtree(i)
        except Exception as e:
            logger.error(e)


if __name__ == '__main__':
    del_file()
    chrak_path()
'''


def models(work_path, project_name):
    return f"""# 模型
import tensorflow as tf
from {work_path}.{project_name}.settings import LR
from {work_path}.{project_name}.settings import N_CLASS
from {work_path}.{project_name}.settings import IMAGE_HEIGHT
from {work_path}.{project_name}.settings import IMAGE_WIDTH
from {work_path}.{project_name}.settings import CAPTCHA_LENGTH
from {work_path}.{project_name}.settings import IMAGE_CHANNALS
from {work_path}.{project_name}.settings import CAPTCHA_CHARACTERS_LENGTH


class Models(object):

    @staticmethod
    def xception_model(fine_ture_at=3):
        covn_base = tf.keras.applications.Xception(include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                                                   pooling='max')
        model = tf.keras.Sequential()
        model.add(covn_base)
        model.add(tf.keras.layers.Dense(512, activation=tf.keras.activations.selu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(256, activation=tf.keras.activations.selu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(4, activation='softmax'))
        covn_base.trainable = False
        for layer in covn_base.layers[:fine_ture_at]:
            layer.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR, amsgrad=True),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['categorical_accuracy'])
        return model

    @staticmethod
    def simple_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS)))
        model.add(tf.keras.layers.Conv2D(16, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(
            tf.keras.layers.Dense(CAPTCHA_CHARACTERS_LENGTH * CAPTCHA_LENGTH, activation=tf.keras.activations.softmax))
        model.add(tf.keras.layers.Reshape((CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH)))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR, amsgrad=True),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['categorical_accuracy'])
        return model

    @staticmethod
    def captcha_model():
        x_input = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS))
        x = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation=tf.keras.activations.relu)(
            x_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Models.identity_block(x, filters=(32, 32, 64), stage=1)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Models.identity_block(x, filters=(64, 64, 128), stage=3)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Models.identity_block(x, filters=(128, 128, 256), stage=5)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        # x = Models.dense_block(x, 32, stage=7)
        # x = Models.dense_block(x, 64, stage=9)
        # x = Models.dense_block(x, 128, stage=11)
        # x = Models.dense_block(x, 256, stage=13)
        x = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(CAPTCHA_CHARACTERS_LENGTH * CAPTCHA_LENGTH, activation=tf.keras.activations.softmax)(
            x)
        x = tf.keras.layers.Reshape((CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH))(x)
        model = tf.keras.Model(inputs=x_input, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR, amsgrad=True),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['acc'])
        return model

    @staticmethod
    def captcha_separableconv2d_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS)))
        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same',
                                         activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(
            tf.keras.layers.Dense(N_CLASS, activation=tf.keras.activations.softmax))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR, amsgrad=True),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['acc'])
        return model

    @staticmethod
    def captcha_DepthwiseConv2D_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS)))
        model.add(tf.keras.layers.DepthwiseConv2D(16, kernel_size=3, strides=(1, 1), padding='same',
                                                  activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.DepthwiseConv2D(32, kernel_size=3, strides=(2, 2), padding='same',
                                                  activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.SpatialDropout2D(0.1))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(tf.keras.layers.SpatialDropout2D(0.1))
        model.add(tf.keras.layers.DepthwiseConv2D(64, kernel_size=3, strides=(1, 1), padding='same',
                                                  activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.DepthwiseConv2D(128, kernel_size=3, strides=(1, 1), padding='same',
                                                  activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
        model.add(tf.keras.layers.SpatialDropout2D(0.1))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(
            tf.keras.layers.Dense(CAPTCHA_CHARACTERS_LENGTH * CAPTCHA_LENGTH, activation=tf.keras.activations.softmax))
        model.add(tf.keras.layers.Reshape((CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH)))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR, amsgrad=True),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['categorical_accuracy'])
        return model

    @staticmethod
    def dense_block(x, units=128, stage=2, block='a'):
        '''
        全连接的残差(短接)
        :param x: 输入层
        :param units: 输出维度空间
        :param stage:名字
        :param block:名字
        :return:层
        '''
        conv_name_base = 'res' + str(stage) + str(block) + '_branch'
        bn_name_base = 'bn' + str(stage) + str(block) + '_branch'
        x_shortcut = x
        x = tf.keras.layers.Dense(units=units, activation=tf.keras.activations.relu, name=conv_name_base)(x)
        x = tf.keras.layers.BatchNormalization(name=bn_name_base)(x)
        x = tf.keras.layers.concatenate([x, x_shortcut])
        return x

    @staticmethod
    def identity_block(x, f=3, filters=(32, 64, 128), stage=2, block='a'):
        '''
        残差块跳跃连接(短接)
        :param x:输入层
        :param filters:列表或元组，长度为3，指定卷积核数量
        :param stage:名字
        :param block:名字
        :param f:卷积核的大小
        :return:层
        '''
        conv_name_base = 'res' + str(stage) + str(block) + '_branch'
        bn_name_base = 'bn' + str(stage) + str(block) + '_branch'
        F1, F2, F3 = filters
        x_shortcut = x
        x = tf.keras.layers.Conv2D(filters=F1, kernel_size=f, strides=1, padding="same", name=conv_name_base + '2a',

                                   activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=1, padding="same", name=conv_name_base + '2b',

                                   activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=1, padding="same", name=conv_name_base + '2c',

                                   activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
        x = tf.keras.layers.concatenate([x, x_shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

    @staticmethod
    def convolutional_block(x, f=3, filters=(32, 64, 128), stage=2, block='a'):
        '''
        残差块跳跃连接
        :param x:输入层
        :param filters:列表或元组，长度为3，指定卷积核数量
        :param stage:名字
        :param block:名字
        :param f:卷积核的大小
        :return:层
        '''
        conv_name_base = 'res' + str(stage) + str(block) + '_branch'
        bn_name_base = 'bn' + str(stage) + str(block) + '_branch'
        F1, F2, F3 = filters
        x_shortcut = x
        x = tf.keras.layers.Conv2D(filters=F1, kernel_size=1, strides=1, padding="same", name=conv_name_base + '2a',
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(0),
                                   activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=1, padding="same", name=conv_name_base + '2b',
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(0),
                                   activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = tf.keras.layers.Conv2D(filters=F2, kernel_size=f, strides=1, padding="same", name=conv_name_base + '2c',
                                   kernel_initializer=tf.keras.initializers.GlorotNormal(0),
                                   activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
        x_shortcut = tf.keras.layers.Conv2D(filters=F3, kernel_size=1, strides=1, padding="same",
                                            name=conv_name_base + '1',
                                            kernel_initializer=tf.keras.initializers.GlorotNormal(0),
                                            activation=tf.keras.activations.relu)(x_shortcut)
        x_shortcut = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x_shortcut)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = tf.keras.layers.concatenate([x, x_shortcut])
        x = tf.keras.layers.ReLU()(x)
        return x

    @staticmethod
    def captcha_12306():
        x_input = tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNALS))
        x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=(1, 1), padding='same',
                                   activation=tf.keras.activations.relu)(x_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Models.identity_block(x=x, f=3, filters=(32, 32, 64), stage=2, block='a')
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Models.identity_block(x=x, f=3, filters=(128, 128, 256), stage=5, block='a')
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = Models.identity_block(x=x, f=3, filters=(256, 256, 512), stage=9, block='a')
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(N_CLASS, activation=tf.keras.activations.softmax)(x)
        model = tf.keras.Model(inputs=x_input, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR, amsgrad=True),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['acc'])
        return model


if __name__ == '__main__':
    model = Models.captcha_model()
    model.summary()
"""


def move_path(work_path, project_name):
    return f'''import random
from {work_path}.{project_name}.settings import train_path
from {work_path}.{project_name}.Function_API import Image_Processing

train_image = Image_Processing.extraction_image(train_path)
random.shuffle(train_image)
Image_Processing.move_path(train_image)
'''


def pack_dataset(work_path, project_name):
    return f"""'''
打包数据
'''
import random
from loguru import logger
from {work_path}.{project_name}.settings import train_path
from {work_path}.{project_name}.settings import validation_path
from {work_path}.{project_name}.settings import test_path
from {work_path}.{project_name}.settings import IMAGE_HEIGHT
from {work_path}.{project_name}.settings import IMAGE_WIDTH
from {work_path}.{project_name}.settings import train_enhance_path
from {work_path}.{project_name}.settings import DATA_ENHANCEMENT
from {work_path}.{project_name}.settings import TFRecord_train_path
from {work_path}.{project_name}.settings import TFRecord_validation_path
from {work_path}.{project_name}.settings import TFRecord_test_path
from {work_path}.{project_name}.Function_API import Image_Processing
from {work_path}.{project_name}.Function_API import WriteTFRecord
from concurrent.futures import ThreadPoolExecutor

if DATA_ENHANCEMENT:
    with ThreadPoolExecutor(max_workers=100) as t:
        for i in Image_Processing.extraction_image(train_path):
            task = t.submit(Image_Processing.preprosess_save_images, i, [IMAGE_HEIGHT, IMAGE_WIDTH])
    train_image = Image_Processing.extraction_image(train_enhance_path)
    random.shuffle(train_image)
    train_lable = Image_Processing.extraction_lable(train_image)
else:
    train_image = Image_Processing.extraction_image(train_path)
    random.shuffle(train_image)
    train_lable = Image_Processing.extraction_lable(train_image)

validation_image = Image_Processing.extraction_image(validation_path)
validation_lable = Image_Processing.extraction_lable(validation_image)

test_image = Image_Processing.extraction_image(test_path)
test_lable = Image_Processing.extraction_lable(test_image)
logger.debug(train_image)
# logger.debug(train_lable)

with ThreadPoolExecutor(max_workers=3) as t:
    t.submit(WriteTFRecord.WriteTFRecord, TFRecord_train_path, train_image, train_lable)
    t.submit(WriteTFRecord.WriteTFRecord, TFRecord_validation_path, validation_image, validation_lable)
    t.submit(WriteTFRecord.WriteTFRecord, TFRecord_test_path, test_image, test_lable)

"""


def rename_suffix(work_path, project_name):
    return f'''from {work_path}.{project_name}.settings import train_path
from {work_path}.{project_name}.Function_API import Image_Processing


Image_Processing.rename_suffix(Image_Processing.extraction_image(train_path))
'''


def save_model(work_path, project_name):
    return f'''import os
import operator
from loguru import logger
from {work_path}.{project_name}.models import Models
from {work_path}.{project_name}.Callback import CallBack
from {work_path}.{project_name}.settings import MODEL
from {work_path}.{project_name}.settings import MODEL_NAME
from {work_path}.{project_name}.settings import model_path
from {work_path}.{project_name}.settings import checkpoint_path

model = operator.methodcaller(MODEL)(Models)
try:
    model.load_weights(os.path.join(checkpoint_path, CallBack.calculate_the_best_weight()))
except:
    raise OSError(f'没有任何的权重和模型在{{model_path}}')
model_save = os.path.join(model_path, MODEL_NAME)
model.save(model_save)
model_path = [model_save]
logger.debug(f'{{model_path}}模型保存成功')
'''


def settings(work_path, project_name):
    return """import os
import datetime

MODE = 'ordinary'

# 学习率
LR = 1e-2

# 训练次数
EPOCHS = 100

# batsh批次
BATCH_SIZE = 64

# 训练多少轮验证损失下不去，学习率/10
LR_PATIENCE = 4

# 训练多少轮验证损失下不去，停止训练
EARLY_PATIENCE = 8

# 分几类
N_CLASS = 80

# 图片高度
IMAGE_HEIGHT = 60

# 图片宽度
IMAGE_WIDTH = 100

# 图片通道
IMAGE_CHANNALS = 3

# 验证码字符集
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_']

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

CAPTCHA_CHARACTERS = number + alphabet + ALPHABET

CAPTCHA_CHARACTERS_LENGTH = len(CAPTCHA_CHARACTERS)

# 验证码的长度
CAPTCHA_LENGTH = 6

# 定义模型的方法,模型在models.py定义
MODEL = 'captcha_model'

# 保存的模型名称
MODEL_NAME = 'captcha.h5'

# 测试的模型名称
MODEL_LEAD_NAME = 'captcha.h5'

# 是否使用数据增强(数据集多的时候不需要用)
DATA_ENHANCEMENT = False

# 是否使用多模型预测
MULITI_MODEL_PREDICTION = False

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
"""


def spider_example(work_path, project_name):
    return '''import time
import base64
import random
import requests
from loguru import logger


def get_captcha():
    r = int(random.random() * 100000000)
    params = {
        'r': str(r),
        's': '0',
    }
    response = requests.get('https://login.sina.com.cn/cgi/pin.php', params=params)
    if response.status_code == 200:
        return response.content


if __name__ == '__main__':
    content = get_captcha()
    if content:
        logger.debug(f'获取验证码成功')
        with open(f'{int(time.time())}.jpg', 'wb') as f:
            f.write(content)
        data = {'img': base64.b64encode(content)}
        response = requests.post('http://127.0.0.1:5006', data=data)
        logger.debug(response.json())
        if response.json().get('return_info') == '处理成功':
            logger.debug(f'验证码为{response.json().get("result")}')
        else:
            logger.error('识别失败')

    else:
        logger.error(f'获取验证码失败')
'''


def sub_filename(work_path, project_name):
    return f'''from {work_path}.{project_name}.settings import train_path
from {work_path}.{project_name}.Function_API import Image_Processing

Image_Processing.rename_path(train_path)
'''


def test_model(work_path, project_name):
    return f'''# 测试模型
import os
import random
import tensorflow as tf
from loguru import logger
from {work_path}.{project_name}.settings import test_path
from {work_path}.{project_name}.settings import model_path
from {work_path}.{project_name}.settings import BATCH_SIZE
from {work_path}.{project_name}.settings import test_pack_path
from {work_path}.{project_name}.settings import MODEL_LEAD_NAME
from {work_path}.{project_name}.settings import MULITI_MODEL_PREDICTION
from {work_path}.{project_name}.Function_API import Image_Processing
from {work_path}.{project_name}.Function_API import Distinguish_image
from {work_path}.{project_name}.Function_API import parse_function

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

test_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(test_pack_path)).map(
    parse_function).batch(BATCH_SIZE)

test_image_list = Image_Processing.extraction_image(test_path)
random.shuffle(test_image_list)

if MULITI_MODEL_PREDICTION:
    model_path = Image_Processing.extraction_image(model_path)
    if not model_path:
        raise OSError(f'{{model_path}}没有模型')
    for i in test_image_list[:10]:
        Distinguish_image.distinguish_images(model_path, i)
else:
    model_path = os.path.join(model_path, MODEL_LEAD_NAME)
    logger.debug(f'加载模型{{model_path}}')
    if not os.path.exists(model_path):
        raise OSError(f'{{model_path}}没有模型')
    for i in test_image_list[:10]:
        Distinguish_image.distinguish_image(model_path, i)
    model = tf.keras.models.load_model(model_path)
    logger.info(model.evaluate(test_dataset))
'''


def train_run(work_path, project_name):
    return f'''import os
import operator
import tensorflow as tf
from loguru import logger
from {work_path}.{project_name}.models import Models
from {work_path}.{project_name}.Callback import CallBack
from {work_path}.{project_name}.settings import MODEL
from {work_path}.{project_name}.settings import EPOCHS
from {work_path}.{project_name}.settings import BATCH_SIZE
from {work_path}.{project_name}.settings import model_path
from {work_path}.{project_name}.settings import MODEL_NAME
from {work_path}.{project_name}.settings import DATA_ENHANCEMENT
from {work_path}.{project_name}.settings import train_path
from {work_path}.{project_name}.settings import train_pack_path
from {work_path}.{project_name}.settings import validation_pack_path
from {work_path}.{project_name}.settings import test_pack_path
from {work_path}.{project_name}.settings import train_enhance_path
from {work_path}.{project_name}.Function_API import cheak_path
from {work_path}.{project_name}.Function_API import Image_Processing
from {work_path}.{project_name}.Function_API import parse_function

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

train_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(train_pack_path)).map(
    parse_function).batch(BATCH_SIZE)

validation_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(validation_pack_path)).map(
    parse_function).batch(
    BATCH_SIZE)

test_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(test_pack_path)).map(
    parse_function).batch(BATCH_SIZE)

model, c_callback = CallBack.callback(operator.methodcaller(MODEL)(Models))

model.summary()

if DATA_ENHANCEMENT:
    logger.debug(f'一共有{{int(len(Image_Processing.extraction_image(train_enhance_path)) / BATCH_SIZE)}}个batch')
else:
    logger.debug(f'一共有{{int(len(Image_Processing.extraction_image(train_path)) / BATCH_SIZE)}}个batch')

model.fit(train_dataset, epochs=EPOCHS, callbacks=c_callback, validation_data=validation_dataset, verbose=2)

save_model_path = cheak_path(os.path.join(model_path, MODEL_NAME))

model.save(save_model_path)

logger.info(model.evaluate(test_dataset))
'''


class New_Work(object):
    def __init__(self, work_path='works', project_name='project'):
        self.work_parh = work_path
        self.project_name = project_name
        self.work = os.path.join(os.getcwd(), work_path)
        if not os.path.exists(self.work):
            os.mkdir(self.work)
        self.path = os.path.join(self.work, self.project_name)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        else:
            raise OSError('已有的项目')

    def file_name(self, name):
        return os.path.join(self.path, name)

    def callback(self):
        with open(self.file_name('Callback.py'), 'w', encoding='utf-8') as f:
            f.write(callback(self.work_parh, self.project_name))

    def app(self):
        with open(self.file_name('app.py'), 'w', encoding='utf-8') as f:
            f.write(app(self.work_parh, self.project_name))

    def captcha_config(self):
        with open(self.file_name('captcha_config.json'), 'w') as f:
            f.write(captcha_config())

    def del_file(self):
        with open(self.file_name('del_file.py'), 'w', encoding='utf-8') as f:
            f.write(del_file(self.work_parh, self.project_name))

    def Function_API(self):
        with open(self.file_name('Function_API.py'), 'w', encoding='utf-8') as f:
            f.write(Function_API(self.work_parh, self.project_name))

    def gen_sample_by_captcha(self):
        with open(self.file_name('gen_sample_by_captcha.py'), 'w', encoding='utf-8') as f:
            f.write(gen_sample_by_captcha(self.work_parh, self.project_name))

    def init_working_space(self):
        with open(self.file_name('init_working_space.py'), 'w', encoding='utf-8') as f:
            f.write(init_working_space(self.work_parh, self.project_name))

    def models(self):
        with open(self.file_name('models.py'), 'w', encoding='utf-8') as f:
            f.write(models(self.work_parh, self.project_name))

    def move_path(self):
        with open(self.file_name('move_path.py'), 'w', encoding='utf-8') as f:
            f.write(move_path(self.work_parh, self.project_name))

    def pack_dataset(self):
        with open(self.file_name('pack_dataset.py'), 'w', encoding='utf-8') as f:
            f.write(pack_dataset(self.work_parh, self.project_name))

    def rename_suffix(self):
        with open(self.file_name('rename_suffix.py'), 'w', encoding='utf-8') as f:
            f.write(rename_suffix(self.work_parh, self.project_name))

    def save_model(self):
        with open(self.file_name('save_model.py'), 'w', encoding='utf-8') as f:
            f.write(save_model(self.work_parh, self.project_name))

    def settings(self):
        with open(self.file_name('settings.py'), 'w', encoding='utf-8') as f:
            f.write(settings(self.work_parh, self.project_name))

    def spider_example(self):
        with open(self.file_name('spider_example.py'), 'w', encoding='utf-8') as f:
            f.write(spider_example(self.work_parh, self.project_name))

    def sub_filename(self):
        with open(self.file_name('sub_filename.py'), 'w', encoding='utf-8') as f:
            f.write(sub_filename(self.work_parh, self.project_name))

    def test_model(self):
        with open(self.file_name('test_model.py'), 'w', encoding='utf-8') as f:
            f.write(test_model(self.work_parh, self.project_name))

    def train_run(self):
        with open(self.file_name('train_run.py'), 'w', encoding='utf-8') as f:
            f.write(train_run(self.work_parh, self.project_name))

    def main(self):
        self.callback()
        self.app()
        self.captcha_config()
        self.del_file()
        self.Function_API()
        self.gen_sample_by_captcha()
        self.init_working_space()
        self.models()
        self.move_path()
        self.pack_dataset()
        self.rename_suffix()
        self.save_model()
        self.settings()
        self.spider_example()
        self.sub_filename()
        self.test_model()
        self.train_run()


if __name__ == '__main__':
    New_Work(work_path='works', project_name='project_yunpian').main()
