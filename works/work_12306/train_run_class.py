import os
import operator
import tensorflow as tf
from loguru import logger
from works.work_12306.models import Models
from works.work_12306.Callback import CallBack
from works.work_12306.settings import MODEL
from works.work_12306.settings import EPOCHS
from works.work_12306.settings import BATCH_SIZE
from works.work_12306.settings import model_path
from works.work_12306.settings import MODEL_NAME
from works.work_12306.settings import DATA_ENHANCEMENT
from works.work_12306.settings import train_path
from works.work_12306.settings import train_pack_path
from works.work_12306.settings import validation_pack_path
from works.work_12306.settings import test_pack_path
from works.work_12306.settings import train_enhance_path
from works.work_12306.Function_API import cheak_path
from works.work_12306.Function_API import Image_Processing
from works.work_12306.Function_API import parse_function

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
    logger.debug(f'一共有{int(len(Image_Processing.extraction_image(train_enhance_path)) / BATCH_SIZE)}个batch')
else:
    logger.debug(f'一共有{int(len(Image_Processing.extraction_image(train_path)) / BATCH_SIZE)}个batch')

model.fit(train_dataset, epochs=EPOCHS, callbacks=c_callback, validation_data=validation_dataset, verbose=2)

save_model_path = cheak_path(os.path.join(model_path, MODEL_NAME))

model.save(save_model_path)

logger.info(model.evaluate(test_dataset))
