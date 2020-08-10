# 函数丢这里
import re
import os
import time
import json
import shutil
import base64
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from loguru import logger
from functools import reduce
import matplotlib.pyplot as plt
from works.weibo_sougou.settings import validation_path
from works.weibo_sougou.settings import test_path
from works.weibo_sougou.settings import MODE
from works.weibo_sougou.settings import N_CLASS
from works.weibo_sougou.settings import IMAGE_HEIGHT
from works.weibo_sougou.settings import IMAGE_WIDTH
from works.weibo_sougou.settings import CAPTCHA_LENGTH
from works.weibo_sougou.settings import IMAGE_CHANNALS
from works.weibo_sougou.settings import CAPTCHA_CHARACTERS_LENGTH
from concurrent.futures import ThreadPoolExecutor


class Image_Processing(object):
    # 分割数据集
    @classmethod
    def move_path(self, path: list, proportion=0.2) -> bool:
        logger.debug(f'数据集有{len(path)},{proportion * 100}%作为验证集,{proportion * 100}%作为测试集')
        division_number = int(len(path) * proportion)
        logger.debug(f'验证集数量为{division_number},测试集数量为{division_number}')
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
    def extraction_image(self, path: str) -> list:
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
        logger.debug(f'开始处理{image_path}')
        image_name = os.path.splitext(os.path.split(image_path)[-1])[0]
        image_suffix = os.path.splitext(os.path.split(image_path)[-1])[-1]
        img_raw = tf.io.read_file(image_path)
        img_tensor = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNALS)
        # img_tensor_up = tf.image.flip_up_down(img_tensor)
        # img_tensor_a = tf.image.resize(img_tensor, size)
        # 旋转
        img_tensor_rotated_90 = tf.image.resize(tf.image.rot90(img_tensor), size)
        img_tensor_rotated_180 = tf.image.resize(tf.image.rot90(tf.image.rot90(img_tensor)), size)
        img_tensor_rotated_270 = tf.image.resize(tf.image.rot90(tf.image.rot90(tf.image.rot90(img_tensor))), size)
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
        image_tensor = [img_tensor_rotated_90, img_tensor_rotated_180, img_tensor_rotated_270, img_tensor_contrast1,
                        img_tensor_contrast9, img_tensor_saturated_1, img_tensor_saturated_9, img_tensor_brightness_1,
                        img_tensor_brightness_4, img_tensor_hue1, img_tensor_hue9, img_tensor_standardization]
        for index, i in tqdm(enumerate(image_tensor), desc='正在生成图片'):
            img_tensor = np.asarray(i.numpy(), dtype='uint8')
            img_tensor = tf.image.encode_jpeg(img_tensor)
            with tf.io.gfile.GFile(f'train_enhance_dataset/{image_name}_{str(index)}{image_suffix}', 'wb') as file:
                file.write(img_tensor.numpy())
        logger.info(f'处理完成{image_path}')
        return True

    @classmethod
    # 展示图片处理后的效果
    def show_image(self, image_path):
        '''
        展示图片处理后的效果
        :param image_path:
        :return:
        '''
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
                raise ValueError(f'有验证码长度大于{CAPTCHA_LENGTH}标签为:{text}')
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
                    raise ValueError(f'字符长度{len(string)}大于设置{CAPTCHA_LENGTH}')
            for i in string:
                lable = dicts.get(i)
                vector_ocr.append(lable)
            # vector_ocr = np.array(vector_ocr)
            return vector_ocr
        else:
            raise ValueError(f'没有mode={mode}提取标签的方法')

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
                # logger.debug(f'一共有{len(dicts)}类')
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
                path = [re.split(divide, i)[0] for i in paths]
                dicts_list = sorted(set(path))
                logger.debug(f'一共有{len(dicts_list)}类')
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
                # logger.debug(f'一共有{len(dicts)}类')
                paths = [os.path.splitext(os.path.split(i)[-1])[0] for i in path_list]
                dicts_list = sorted(set(paths))
                # dicts_list = sorted(set(path))
                logger.debug(f'一共有{len(dicts_list)}类')
                dicts = dict((name, index) for index, name in enumerate(dicts_list))
                d = dict((index, name) for index, name in enumerate(dicts_list))
                with open('n_class.json', 'w', encoding='utf-8') as f:
                    f.write(json.dumps(d, ensure_ascii=False))
                lable_list = [self.text2vector(dicts.get(i), mode=mode) for i in paths]
                return lable_list
        elif mode == 'ordinary_ocr':
            if suffix:
                # dicts = self.extraction_ocr_dict(path_list, divide=divide)
                # logger.debug(f'一共有{len(dicts)}类')
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
        else:
            raise ValueError(f'没有mode={mode}提取标签的方法')


# 打包数据
class WriteTFRecord(object):
    @classmethod
    def WriteTFRecord(self, TFRecord_path, datasets: list, lables: list, file_name='dataset.tfrecords', mode=MODE):
        num_count = len(datasets)
        lables_count = len(lables)
        if not os.path.exists(TFRecord_path):
            os.mkdir(TFRecord_path)
        logger.info(f'文件个数为:{num_count}')
        logger.info(f'标签个数为:{lables_count}')
        filename = os.path.join(TFRecord_path, file_name)
        writer = tf.io.TFRecordWriter(filename)
        logger.info(f'开始保存{filename}')
        for dataset, lable in zip(datasets, lables):
            image_bytes = open(dataset, 'rb').read()
            num_count = num_count - 1
            logger.debug(f'剩余{num_count}图片待打包')
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                             'lable': tf.train.Feature(float_list=tf.train.FloatList(value=lable))}))
            # 序列化
            serialized = example.SerializeToString()
            writer.write(serialized)
        logger.info(f'保存{filename}成功')
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
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'lable': tf.io.FixedLenFeature((80,), tf.float32)
    }
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
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'lable': tf.io.FixedLenFeature([CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH], tf.float32)
        }
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        lable_tensor = parsed_example['lable']
        return (img_tensor, lable_tensor)
    elif mode == 'n_class':
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'lable': tf.io.FixedLenFeature([N_CLASS], tf.float32)
        }
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        lable_tensor = parsed_example['lable']
        return (img_tensor, lable_tensor)
    elif mode == 'ordinary_ocr':
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'lable': tf.io.FixedLenFeature([CAPTCHA_LENGTH, N_CLASS], tf.float32)
        }
        parsed_example = tf.io.parse_single_example(exam_proto, features)
        img_tensor = tf.image.decode_jpeg(parsed_example['image'], channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = img_tensor / 255.
        lable_tensor = parsed_example['lable']
        return (img_tensor, lable_tensor)
    else:
        raise ValueError(f'没有mode={mode}映射的方法')


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
            raise ValueError(f'有验证码长度大于{CAPTCHA_LENGTH}标签为:{text}')
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
                recognition_rate.append({chr(char_code): '%.2f' % (recognition * 100) + '%'})
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
        logger.info(f'预测值为{lable_forecast.replace("_", "")},真实值为{lable_real.replace("_", "")}')
        logger.info(f'每个字符的识别率:{recognition_rate}')
        logger.info(f'整体识别率{overall_recognition_rate}')
        if str(lable_forecast.replace("_", "")) != str(lable_real.replace("_", "")):
            logger.error(f'预测失败的图片路径为:{jpg_path}')
            self.true_value = self.true_value + 1
            logger.debug(f'正确率:{(self.predicted_value / self.true_value) * 100}%')
            if self.predicted_value > 0:
                logger.debug(f'预测正确{self.predicted_value}张图片')
        else:
            self.predicted_value = self.predicted_value + 1
            self.true_value = self.true_value + 1
            logger.debug(f'正确率:{(self.predicted_value / self.true_value) * 100}%')
            if self.predicted_value > 0:
                logger.debug(f'预测正确{self.predicted_value}张图片')
        return lable_forecast

    # 预测方法
    @classmethod
    def distinguish_image(self, model_path, jpg_path, suffix=True, divide='_', mode=MODE):
        if mode == 'ordinary':
            model = tf.keras.models.load_model(model_path)
            forecast = model.predict(Image_Processing.load_image(jpg_path))
            overall_recognition_rate, recognition_rate, lable_forecast = self.vector2text(forecast[0])
            lable_real = self.extraction_lable_name(jpg_path, suffix, divide)
            logger.info(f'预测为{lable_forecast.replace("_", "")},真实为{lable_real.replace("_", "")}')
            logger.info(f'每个字符的识别率:{recognition_rate}')
            logger.info(f'整体识别率{overall_recognition_rate}')
            if str(lable_forecast.replace("_", "")) != str(lable_real.replace("_", "")):
                logger.error(f'预测失败的图片路径为:{jpg_path}')
                self.true_value = self.true_value + 1
                logger.debug(f'正确率:{(self.predicted_value / self.true_value) * 100}%')
                if self.predicted_value > 0:
                    logger.debug(f'预测正确{self.predicted_value}张图片')
            else:
                self.predicted_value = self.predicted_value + 1
                self.true_value = self.true_value + 1
                logger.debug(f'正确率:{(self.predicted_value / self.true_value) * 100}%')
                if self.predicted_value > 0:
                    logger.debug(f'预测正确{self.predicted_value}张图片')
            return lable_forecast
        elif mode == 'n_class':
            model = tf.keras.models.load_model(model_path)
            forecast = model.predict(Image_Processing.load_image(jpg_path))
            lable_real = self.extraction_lable_name(jpg_path, suffix, divide)
            recognition_rate, lable_forecast = Distinguish_image.vector2lable_name(forecast[0])
            logger.info(f'预测值为{lable_forecast.replace("_", "")},真实值为{lable_real.replace("_", "")}')
            logger.info(f'图片的识别率:{recognition_rate}')
            if str(lable_forecast.replace("_", "")) != str(lable_real.replace("_", "")):
                logger.error(f'预测失败的图片路径为:{jpg_path}')
                self.true_value = self.true_value + 1
                logger.debug(f'正确率:{(self.predicted_value / self.true_value) * 100}%')
                if self.predicted_value > 0:
                    logger.debug(f'预测正确{self.predicted_value}张图片')
            else:
                self.predicted_value = self.predicted_value + 1
                self.true_value = self.true_value + 1
                logger.debug(f'正确率:{(self.predicted_value / self.true_value) * 100}%')
                if self.predicted_value > 0:
                    logger.debug(f'预测正确{self.predicted_value}张图片')
            return lable_forecast
        elif mode == 'ordinary_ocr':
            pass
        else:
            raise ValueError(f'没有{mode}的预测方法')

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
        vertor = []
        with ThreadPoolExecutor() as t:
            for model in model_path:
                result = t.submit(self.model_app, model, jpg)
                vertor.append(result.result())
        vertor_sum = reduce(lambda x, y: x + y, vertor)
        forecast = vertor_sum[0] / len(vertor)
        overall_recognition_rate, recognition_rate, lable_forecast = self.vector2text(forecast)
        return (overall_recognition_rate, recognition_rate, lable_forecast.replace("_", ""))


def cheak_path(path):
    while True:
        if os.path.exists(path):
            paths, name = os.path.split(path)
            name, mix = os.path.splitext(name)
            name = name + f'_{int(time.time())}'
            path = os.path.join(paths, name + mix)
        if not os.path.exists(path):
            return path
