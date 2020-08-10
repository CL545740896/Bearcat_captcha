# 打包数据
import random
from loguru import logger
from works.project_12306_ocr.settings import train_path
from works.project_12306_ocr.settings import validation_path
from works.project_12306_ocr.settings import test_path
from works.project_12306_ocr.settings import IMAGE_HEIGHT
from works.project_12306_ocr.settings import IMAGE_WIDTH
from works.project_12306_ocr.settings import train_enhance_path
from works.project_12306_ocr.settings import DATA_ENHANCEMENT
from works.project_12306_ocr.settings import TFRecord_train_path
from works.project_12306_ocr.settings import TFRecord_validation_path
from works.project_12306_ocr.settings import TFRecord_test_path
from works.project_12306_ocr.Function_API import Image_Processing
from works.project_12306_ocr.Function_API import WriteTFRecord
from concurrent.futures import ThreadPoolExecutor

if DATA_ENHANCEMENT:
    with ThreadPoolExecutor(max_workers=100) as t:
        for i in Image_Processing.extraction_image(train_path):
            task = t.submit(Image_Processing.preprosess_save_images, i, [IMAGE_HEIGHT, IMAGE_WIDTH])
    train_image = Image_Processing.extraction_image(train_enhance_path)
    Image_Processing.save_ocr_class(train_image)
    random.shuffle(train_image)
    train_lable = Image_Processing.extraction_ocr_lable(train_image)
else:
    train_image = Image_Processing.extraction_image(train_path)
    Image_Processing.save_ocr_class(train_image)
    random.shuffle(train_image)
    train_lable = Image_Processing.extraction_ocr_lable(train_image)

validation_image = Image_Processing.extraction_image(validation_path)
validation_lable = Image_Processing.extraction_ocr_lable(validation_image)

test_image = Image_Processing.extraction_image(test_path)
test_lable = Image_Processing.extraction_ocr_lable(test_image)
# logger.debug(train_image)
# logger.debug(train_lable)

with ThreadPoolExecutor(max_workers=3) as t:
    t.submit(WriteTFRecord.WriteTFRecord_verification, TFRecord_train_path, train_image, train_lable)
    t.submit(WriteTFRecord.WriteTFRecord_verification, TFRecord_validation_path, validation_image, validation_lable)
    t.submit(WriteTFRecord.WriteTFRecord_verification, TFRecord_test_path, test_image, test_lable)
