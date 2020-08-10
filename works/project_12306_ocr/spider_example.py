import time
import base64
import random
import requests
from loguru import logger


def get_captcha():
    r = int(random.random() * 100000000)
    params = {
        'r': str(r),
        's': '0',
    }
    response = requests.get('https://login.sina.com.cn/cgi/pin.php', params=params)
    if response.status_code == 200:
        return response.content


if __name__ == '__main__':
    content = get_captcha()
    if content:
        logger.debug(f'获取验证码成功')
        with open(f'{int(time.time())}.jpg', 'wb') as f:
            f.write(content)
        data = {'img': base64.b64encode(content)}
        response = requests.post('http://127.0.0.1:5006', data=data)
        logger.debug(response.json())
        if response.json().get('return_info') == '处理成功':
            logger.debug(f'验证码为{response.json().get("result")}')
        else:
            logger.error('识别失败')

    else:
        logger.error(f'获取验证码失败')
