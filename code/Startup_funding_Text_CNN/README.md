## Startup Funding success Classification Model (using TextCNN) implemented with PyTorch

### 项目简介
本项目基于TextCNN，学习了初创公司的描述文本语义特征，然后通过全连接网络和初创公司的基本特征进行融合出来，最后得出初创公司是否成功获得融资。
此处代码借鉴以下博客内容进行了修改得到。
**TextCNN**: Kim, Yoon "Convolutional Neural Networks for Sentence Classification." Proceedings of EMNLP. 2014.
**论文地址**：[https://www.aclweb.org/anthology/D14-1181.pdf](https://www.aclweb.org/anthology/D14-1181.pdf)
**详细介绍**：[论文复现TextCNN(基于PyTorch) - 简书](https://www.jianshu.com/p/ed0a82780c20)


### 运行代码


**进入主项目目录**

`cd SentenceClassfication`

**解压数据集**

`unzip data.zip`


**进入TextCNN目录**

`cd TextCNN`
    
**安装依赖**

`pip install -r requirements.txt`

**预处理**

`python preprocess.py`

默认为SU1数据集(初创公司融资不平衡数据，SU2为采样后的平衡数据)，运行下列命令可以处理SU1数据集
'python preprocess.py --dataset SU1'

**运行训练+测试**
 
`python main_fused.py`
运行完毕之后，即可得到测试集的效果。


`config.yaml`中可以修改相应配置，实现不同数据集的预测

