from works.weibo_sougou.settings import train_path
from works.weibo_sougou.Function_API import Image_Processing


Image_Processing.rename_suffix(Image_Processing.extraction_image(train_path))
