# 增强后文件太多，手动删非常困难，直接用代码删
import shutil
from tqdm import tqdm
from loguru import logger
from works.project_12306_ocr.settings import train_path
from works.project_12306_ocr.settings import test_path
from works.project_12306_ocr.settings import validation_path
from works.project_12306_ocr.settings import train_enhance_path
from works.project_12306_ocr.settings import train_pack_path
from works.project_12306_ocr.settings import validation_pack_path
from works.project_12306_ocr.settings import test_pack_path
from concurrent.futures import ThreadPoolExecutor


def del_file(path):
    try:
        shutil.rmtree(path)
        logger.debug(f'成功删除{path}')
    except WindowsError as e:
        logger.error(e)


if __name__ == '__main__':
    path = [train_path, test_path, validation_path, train_enhance_path, train_pack_path, validation_pack_path,
            test_pack_path]
    with ThreadPoolExecutor(max_workers=7) as t:
        for i in tqdm(path, desc='正在删除'):
            t.submit(del_file, i)
