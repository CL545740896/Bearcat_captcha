# 熊猫不定长验证码识别

## 本自述文件会自述以下内容
##### 1.项目环境安装与启动
        1.1 GPU环境安装
        1.2 项目启动
##### 2.项目结构描述
        2.1项目结构描述
##### 3.识别验证码的思路

***注意任何时候你都应该备份你的数据集，数据集来之不易***

***注意任何时候你都应该备份你的数据集，数据集来之不易***

***注意任何时候你都应该备份你的数据集，数据集来之不易***

# 1.项目环境安装与启动
## 1.1 GPU环境安装
本项目在tensorflow2.1或2.2下面都可以运行

但是两种的安装方法都有区别下面详细说一下(windowns环境):

拉取项目

git clone https://gitclone.com/github.com/yuzhiyizhan/Bearcat_captcha

git clone https://github.com/yuzhiyizhan/Bearcat_captcha

CPU的直接命令行 

pip install tensorflow==2.2 -i https://pypi.douban.com

然后

pip install -r requirements.txt -i https://pypi.douban.com/simple

    tensorflow2.1
    1.安装CUDA 10版本 (官网)[https://developer.nvidia.com/cuda-toolkit]
    2.由于CUDA会自动配好环境本项目不在详述 在命令行输入 nvcc -V 查看CUDA版本
    3.安装conda (推荐在清华镜像站下载Anaconda或者Miniconda都可以)
    4.更新一下conda (conda update -n base conda)
    5.创建python3.7.7的虚拟环境并进入 (conda create -n example python=3.7.7) (conda activate example)
    6.安装tensorflow2.1 (conda install tensorflow-gpu==2.1)
    7.再安装其他依赖 (pip install -r requirements.txt -i https://pypi.douban.com/simple)
    
    tensorflow2.2
    1.安装CUDA 11版本 (官网)[https://developer.nvidia.com/cuda-toolkit]
    2.由于CUDA会自动配好环境本项目不在详述 在命令行输入 nvcc -V 查看CUDA版本
    3.安装conda (推荐在清华镜像站下载Anaconda或者Miniconda都可以)
    4.更新一下conda (conda update -n base conda)
    5.创建python3.7.7的虚拟环境并进入 (conda create -n example python=3.7.7) (conda activate example)
    6.安装tensorflow2.2 (pip install tensorflow-gpu==2.2 -i https://pypi.douban.com/simple)
    7.安装cudnn (conda install cudatoolkit=10.1 cudnn=7.6.5)
    8.再安装其他依赖 (pip install -r requirements.txt -i https://pypi.douban.com/simple)

## 1.2 项目启动

### 项目基于tensorflow2.1(2.2也可以)
    项目的输入图片的格式为.jpg
    不是.jpg后缀也不用慌本项目有修改后缀的代码
    后面会介绍

### 项目启动
    ps:不想自己练的直接运行app.py
    默认开启5006端口,post请求接受一个参数img
    需要base64一下,具体请看spider_example.py
    
### 第0步:新建项目

    运行New_work.py

### 第一步:初始化工作路径

    运行init_working_space.py
    python init_working_space.py

### 第二步:准备标注好的数据(注意数据不能放太深,一个文件夹下面就放上数据)(微博加搜狗验证码，其他验证码需自行修改)

    1.将训练数据放到train_dataset文件夹

    2.将验证数据放到validation_dataset文件夹

    3.将测试数据放到test_dataset文件夹

### 如果你的标注数据是一坨的话按照下面步骤区分开来

    1.将一坨数据放到train_dataset文件夹

    2.运行move_path.py
      python move_path.py

### 如果你暂时没有数据,不用慌,先用生成的数据集吧

    运行gen_sample_by_captcha.py

### 第三步:修改配置文件

    这个后面在详细说先用默认设置启动项目吧

### 第四步:打包数据

    运行pack_dataset.py
    python pack_dataset.py

### 第五步:添加模型(model)

    暂时先使用项目自带的模型吧

### 第六步:编译模型

    暂时先默认吧

### 第七步:开始训练

    运行train_run.py
    python train_run.py
    
### 第八步:开启可视化

    tensorboard --logdir "logs"

### 第九步:评估模型

    丹药出来后要看一下是几品丹药
    运行test_model.py

### 第十步:开启后端

    运行app.py
    python app.py
    
### 第十一步:调用接口

    先运行本项目给的例子感受一下
    python spider_example.py

##下面开始补充刚刚省略的一些地方,由于设置文件备注比较完善，解释部分参数

### MODE
    目前一共三种
    'ordinary'      微博加搜狗
    'n_class'       12306图片
    'ordinary_ocr'  12306文字
    
    

### 是否使用数据增强(数据集多的时候不需要用)
    DATA_ENHANCEMENT = False
数据集不够或者过拟合时，可以考虑数据增强下

增强方法在Function_API.py里面的Image_Processing.preprosess_save_images

### 验证码的长度
    CAPTCHA_LENGTH = 6
    
这个数字要取你要识别验证码的最大长度,不足的会自动用'_'补齐

否则会报错raise ValueError

### BATCH_SIZE

    BATCH_SIZE = 32

如果你的显卡很牛逼，可以尝试调大点

### 训练次数

    EPOCHS = 100

请放心调有训练多少轮验证损失下不去，停止训练的回调设置

还有断点续训的回调设置

    EARLY_PATIENCE = 8
    
定义模型的方法名字,模型在models.py里的Model类 (一个方法就是一个网络)

    MODEL = 'captcha_model'

是否使用多模型预测 (人多力量大奥利给)

    MULITI_MODEL_PREDICTION = False

其他设置如果没有特别情况，尽量不要改

# 2.项目结构描述

## 2.1项目结构描述

## 文件夹

### works
    工作目录

### App_model
    后端模型保存路径

### checkpoint
    保存检查点
    
### CSVLogger
    把训练轮结果数据流到 csv 文件

### logs
    保存被 TensorBoard 分析的日志文件

### model
    保存模型
    
### train_dataset
    保存训练集
    
### train_enhance_dataset
    保存增强后的训练集
    
### train_pack_dataset
    保存打包好的训练集
    
### validation_dataset
    保存验证集
    
### vailidation_pack_dataset
    保存打包好的验证集
    
### test_dataset
    保存测试集
    
### test_pack_dataset
    保存打包好的测试集
    
## 文件

### New_work.py
    新建工作目录

### app.py
    开启后端

### Callback.py
    回调函数参考
    [keras中文官网](https://keras.io/zh/callbacks/)
    运行该文件会返回一个最佳的权重文件
    
### captcha_config.json
    生成验证码的配置文件
      "image_suffix": "jpg",生成验证码的后缀
      "count": 20000,生成验证码的数量
      "char_count": [4, 5, 6],生成验证码的长度
      "width": 100,生成验证码的宽度
      "height": 60，生成验证码的高度

### del_file.py
    删除所有数据集的文件
    这里是防止数据太多手动删不动
    
### Function_API.py
    项目核心，三大类
    Image_Processing
    图片处理和标签处理
    WriteTFRecord
    打包数据集
    Distinguish_image
    预测类模型生成后用这个类来预测和部署
    
### gen_sample_by_captcha.py
    生成验证码
    
### init_working_space.py
    初始化工作目录
    ***注意:此文件只在第一次运行项目时运行***
    ***因为这会重置checkpoint CSVLogger logs***
    
### models.py
    搭建模型网络
    
### pack_dataset.py
    打包数据集
  
### rename_suffix.py
    修改训练集文件为.jpg后缀
    验证集文件和测试集文件有需要修改后缀自行改代码
    
### settings.py
    项目的设置文件
    
### spider_example.py
    爬虫调用例子
    返回return_code状态码
    return_info处理状态
    result识别结果
    recognition_rate每个字符的识别率
    overall_recognition_rate最低识别率
    
### sub_filename.py
    替换文件名
    例如文件名为test.01.jpg
    运行后会修改为
    test_01.jpg

### test_model.py
    读取模型进行测试

### train_run.py
    开始训练

### Adjust_parameters.py
    测试中的功能,超参数搜索还没完善

### save_model.py
    把最好的权重保存成模型
    以正确率命名

# 3.识别验证码的思路
我们知道输入神经网络都是张量

那么我们看看图片的张量是怎么样子的

tf.Tensor(
[[[[1.        ]
   [1.        ]
   [1.        ]
   ...
   [1.        ]
   [1.        ]
   [1.        ]]

  [[1.        ]
   [1.        ]
   [1.        ]
   ...
   [1.        ]
   [1.        ]
   [1.        ]]

  [[1.        ]
   [1.        ]
   [1.        ]
   ...
   [1.        ]
   [1.        ]
   [1.        ]]

  ...

  [[0.8980392 ]
   [1.        ]
   [1.        ]
   ...
   [0.90588236]
   [1.        ]
   [0.8901961 ]]

  [[1.        ]
   [1.        ]
   [1.        ]
   ...
   [1.        ]
   [0.92156863]
   [1.        ]]

  [[0.9882353 ]
   [0.95686275]
   [1.        ]
   ...
   [0.91764706]
   [1.        ]
   [0.99607843]]]], shape=(1, 40, 100, 1), dtype=float32)

这是经过 本项目 Image_Processing.load_image 处理后的图片张量的样子

    def load_image(self, path):
        '''
        预处理图片函数
        :param path:图片路径
        :return: 处理好的路径
        '''
        img_raw = tf.io.read_file(path)
        # channel 是彩色图片
        img_tensor = tf.image.decode_jpeg(img_raw, channels=IMAGE_CHANNALS)
        img_tensor = tf.image.resize(img_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor = img_tensor / 255.
        img_tensor = tf.expand_dims(img_tensor, 0)
        return img_tensor
        
已经调整好形状并归一化了

那么标签呢？

首先说说独热编码是怎么回事

例如两个动物猫和狗:

那么表示猫我们用 [1,0]

那么表示狗我们用 [0,1]

这就是独热编码了，为了方便说明和理解我使用数字0到9说明一下标签的处理

例如我们要识别长度为4的验证码 有一张验证码的标签为 5206 那么标签要处理成

[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,  |  0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,  |  1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,  |  0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,]
        
为了方便查看我用  |  隔开了 实际中要去掉

可以看到

0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,

表示的就是5，那么其他数字依此类推

那么一张验证码为520的怎么处理呢？

[0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,  |  0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,  |  1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,  |  0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,]

可以看到

0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,

表示为5，多出的0.是表示空白字符

0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,

当我们识别有空白字符时，说明验证码长度不为4，后面把空白字符去掉即可，本项目用'_'代表空白字符

后面就是打包和训练了

## 关于12306验证码识别的想法

通过抓包可以知道验证码文字部分在图片的上方

验证码图片部分有6张图片且图片的分布是固定的也就是说坐标是固定的

那么可以把图片分割成9份，分开来识别，当然现在只是想想肯定有更好的思路

特别感谢下面一些项目对我的启发

[安师大教务系统验证码检测](https://github.com/AHNU2019/AHNU_captcha)

[cnn_captcha](https://github.com/nickliqian/cnn_captcha)

[captcha_trainer](https://github.com/kerlomz/captcha_trainer)

[captcha-weibo](https://github.com/skygongque/captcha-weibo/blob/master/client.py)

感谢大佬们的数据集让我省去很多成本和时间

搜狗验证码链接：https://pan.baidu.com/s/13wMK3GXaTZ-yaX0vNDG7Ww 提取码：9uxv
-------------------------------------
作者: kerlomz
来源: 夜幕爬虫安全论坛
原文链接: https://bbs.nightteam.cn/thread-149.htm
版权声明: 若无额外声明，本帖为作者原创帖，转载请附上帖子链接！

微博验证码链接：https://pan.baidu.com/s/1w5-MMzX47US3GS8a7xlSBw 提取码: 74uv
-------------------------------------
作者: kerlomz
来源: 夜幕爬虫安全论坛
原文链接: https://bbs.nightteam.cn/thread-470.htm
版权声明: 若无额外声明，本帖为作者原创帖，转载请附上帖子链接！

12306验证码链接：https://pan.baidu.com/s/1SFflCdfKmI6UW1E12GErOg 提取码：e89o
-------------------------------------
作者: sml2h3
来源: 夜幕爬虫安全论坛
原文链接: https://bbs.nightteam.cn/thread-84.htm
版权声明: 若无额外声明，本帖为作者原创帖，转载请附上帖子链接！

### ***此项目以研究学习为目的，禁止用于非法用途***
### 再次说明项目的tensorflow的版本是2.1(2.2)不要搞错了

### 经过我的训练，微博加搜狗的验证码正确率达到了97.54%

### 模型保存在App_model文件夹里,与大家共同学习

### ps:新手上路，轻喷
### 如果觉得我写的不好或者想教我CRNN + CTC 或者有不懂的地方

### 加我的微信

qq2387301977
 
### 备注熊猫验证

# 更新日志

## 2020/08/09

    微博加搜狗验证码识别率99.75%
    
    12306图片识别率99.46%

    待更新12306文字
   
## 2020/08/10
    
    12306文字识别率99.7%
    
    待更新整合api
    
## 2020/08/11
    
    整合API
    
    添加MODE设置
    'ordinary'      微博加搜狗
    'n_class'       12306图片
    'ordinary_ocr'  12306文字
    
    待更新模型部署
    