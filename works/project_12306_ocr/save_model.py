import os
import operator
from loguru import logger
from works.project_12306_ocr.models import Models
from works.project_12306_ocr.Callback import CallBack
from works.project_12306_ocr.settings import MODEL
from works.project_12306_ocr.settings import MODEL_NAME
from works.project_12306_ocr.settings import model_path
from works.project_12306_ocr.settings import checkpoint_path

model = operator.methodcaller(MODEL)(Models)
try:
    model.load_weights(os.path.join(checkpoint_path, CallBack.calculate_the_best_weight()))
except:
    raise OSError(f'没有任何的权重和模型在{model_path}')
model_save = os.path.join(model_path, MODEL_NAME)
model.save(model_save)
model_path = [model_save]
logger.debug(f'{model_path}模型保存成功')
