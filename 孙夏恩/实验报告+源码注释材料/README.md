## 实验目录结构如下
```
ProjectDir
├── dev.txt
├── train.txt
├── train
│   ├── flac
|       ├── ...
|
├──dev
|   ├──flac
|   |  ├──...
|
├──eval
|   ├──flac
|   |  ├──...
|
├──code&annotations.ipynb
├──code&annotations.html（ipynb的html版本，方便查看）
```
## 实验环境
python3.7
需要拥有以下库依赖

必选：

librosa
tensorflow
keras
numpy
spafe

可选：

pandas
matplotlib
seaborn

## 实验运行
使用jupyterlab、jupyternotebook打开code&annotations.ipynb依次阅读代码模块标题，按顺序运行即可实现训练、验证、评测并保存结果等实验要求
