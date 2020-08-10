# 检查项目路径
import os
import shutil
from tqdm import tqdm
from loguru import logger
from works.weibo_sougou.settings import App_model_path
from works.weibo_sougou.settings import checkpoint_path


def chrak_path():
    path = os.getcwd()
    paths = ['test_dataset', 'train_dataset', 'validation_dataset', 'train_enhance_dataset', 'train_pack_dataset',
             'validation_pack_dataset', 'test_pack_dataset', 'model', 'logs', 'CSVLogger', checkpoint_path,
             App_model_path]
    for i in tqdm(paths, desc='正在创建文件夹'):
        mix = os.path.join(path, i)
        if not os.path.exists(mix):
            os.mkdir(mix)


def del_file():
    path = [os.path.join(os.getcwd(), 'CSVLogger'),
            os.path.join(os.getcwd(), 'logs'), checkpoint_path]
    for i in tqdm(path, desc='正在删除'):
        try:
            shutil.rmtree(i)
        except Exception as e:
            logger.error(e)


if __name__ == '__main__':
    del_file()
    chrak_path()
