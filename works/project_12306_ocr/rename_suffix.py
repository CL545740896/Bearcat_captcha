from works.project_12306_ocr.settings import train_path
from works.project_12306_ocr.Function_API import Image_Processing


Image_Processing.rename_suffix(Image_Processing.extraction_image(train_path))
