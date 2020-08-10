import random
from works.weibo_sougou.settings import train_path
from works.weibo_sougou.Function_API import Image_Processing

train_image = Image_Processing.extraction_image(train_path)
random.shuffle(train_image)
Image_Processing.move_path(train_image)
