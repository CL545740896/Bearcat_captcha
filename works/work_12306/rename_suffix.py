from works.work_12306.settings import train_path
from works.work_12306.Function_API import Image_Processing


Image_Processing.rename_suffix(Image_Processing.extraction_image(train_path))
