# 测试模型
import os
import random
import tensorflow as tf
from loguru import logger
from works.project_12306_ocr.settings import test_path
from works.project_12306_ocr.settings import model_path
from works.project_12306_ocr.settings import BATCH_SIZE
from works.project_12306_ocr.settings import test_pack_path
from works.project_12306_ocr.settings import MODEL_LEAD_NAME
from works.project_12306_ocr.settings import MULITI_MODEL_PREDICTION
from works.project_12306_ocr.Function_API import Image_Processing
from works.project_12306_ocr.Function_API import Distinguish_image
from works.project_12306_ocr.Function_API import parse_function_verification

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

test_dataset = tf.data.TFRecordDataset(Image_Processing.extraction_image(test_pack_path)).map(
    parse_function_verification).batch(BATCH_SIZE)

test_image_list = Image_Processing.extraction_image(test_path)
random.shuffle(test_image_list)

if MULITI_MODEL_PREDICTION:
    model_path = Image_Processing.extraction_image(model_path)
    if not model_path:
        raise OSError(f'{model_path}没有模型')
    for i in test_image_list[:10]:
        Distinguish_image.distinguish_images(model_path, i)
else:
    model_path = os.path.join(model_path, MODEL_LEAD_NAME)
    logger.debug(f'加载模型{model_path}')
    if not os.path.exists(model_path):
        raise OSError(f'{model_path}没有模型')
    for i in test_image_list[:10]:
        Distinguish_image.distinguish_image(model_path, i)
    model = tf.keras.models.load_model(model_path)
    logger.info(model.evaluate(test_dataset))
