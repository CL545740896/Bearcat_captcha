import os
import json
import operator
import tensorflow as tf
from flask import Flask
from flask import request
from loguru import logger
from works.project_12306_ocr.models import Models
from works.project_12306_ocr.Callback import CallBack
from works.project_12306_ocr.settings import MODEL
from works.project_12306_ocr.settings import MODEL_NAME
from works.project_12306_ocr.settings import checkpoint_path
from works.project_12306_ocr.settings import App_model_path
from works.project_12306_ocr.settings import MULITI_MODEL_PREDICTION
from works.project_12306_ocr.Function_API import Distinguish_image
from works.project_12306_ocr.Function_API import Image_Processing

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)

app = Flask(__name__)
if os.listdir(App_model_path):
    model_path = Image_Processing.extraction_image(App_model_path)
    logger.debug(f'{model_path}模型加载成功')
else:
    model = operator.methodcaller(MODEL)(Models)
    try:
        model.load_weights(os.path.join(checkpoint_path, CallBack.calculate_the_best_weight()))
    except:
        raise OSError(f'没有任何的权重和模型在{App_model_path}')
    model_save = os.path.join(App_model_path, MODEL_NAME)
    model.save(model_save)
    model_path = [model_save]
    logger.debug(f'{model_path}模型加载成功')


# logger.debug(model_path)

@app.route("/", methods=['POST'])
def captcha_predict():
    return_dict = {'return_code': '200', 'return_info': '处理成功', 'result': False, 'recognition_rate': 0,
                   'overall_recognition_rate': 0}
    get_data = request.form.to_dict()
    if 'img' in get_data.keys():
        base64_str = request.form['img']
        try:
            if MULITI_MODEL_PREDICTION:
                overall_recognition_rate, recognition_rate, lable_forecast = Distinguish_image.distinguish_apis(
                    model_path=model_path,
                    base64_str=base64_str)
                return_dict['result'] = lable_forecast
                return_dict['recognition_rate'] = recognition_rate
                return_dict['overall_recognition_rate'] = overall_recognition_rate
            else:
                overall_recognition_rate, recognition_rate, lable_forecast = Distinguish_image.distinguish_api(
                    model_path=model_path,
                    base64_str=base64_str)
                return_dict['result'] = lable_forecast
                return_dict['recognition_rate'] = recognition_rate
                return_dict['overall_recognition_rate'] = overall_recognition_rate
        except Exception as e:
            return_dict['result'] = str(e)
            return_dict['return_info'] = '模型识别错误'
    else:
        return_dict['return_code'] = '5004'
        return_dict['return_info'] = '参数错误，没有img属性'
    return json.dumps(return_dict, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)
