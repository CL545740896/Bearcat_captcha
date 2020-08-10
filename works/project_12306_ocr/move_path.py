import random
from works.project_12306_ocr.settings import train_path
from works.project_12306_ocr.Function_API import Image_Processing

train_image = Image_Processing.extraction_image(train_path)
random.shuffle(train_image)
Image_Processing.move_path(train_image)
