import os
import operator
from loguru import logger
from works.work_12306.models import Models
from works.work_12306.Callback import CallBack
from works.work_12306.settings import MODEL
from works.work_12306.settings import MODEL_NAME
from works.work_12306.settings import model_path
from works.work_12306.settings import checkpoint_path

model = operator.methodcaller(MODEL)(Models)
try:
    model.load_weights(os.path.join(checkpoint_path, CallBack.calculate_the_best_weight()))
except:
    raise OSError(f'没有任何的权重和模型在{model_path}')
model_save = os.path.join(model_path, MODEL_NAME)
model.save(model_save)
model_path = [model_save]
logger.debug(f'{model_path}模型保存成功')
