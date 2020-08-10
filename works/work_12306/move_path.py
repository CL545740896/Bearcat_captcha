import random
from works.work_12306.settings import train_path
from works.work_12306.Function_API import Image_Processing

train_image = Image_Processing.extraction_image(train_path)
random.shuffle(train_image)
Image_Processing.move_path(train_image)
