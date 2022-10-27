## cs231n
已完结。
水平一般，谨慎参考。

### 本地运行tips/奇怪的坑

google.colab不必安装，可以手动注释掉每个.ipynb的第一个代码块，本机能运行jupyter notebook即可。

数据集下载：在 cs231n/datasets 下用编辑器打开.sh文件，根据其中链接手动下载到文件夹下并解压。

assignment3 Self_Supervised_Learning.ipynb中所有训练块若无法正常训练，可尝试将调用的所有DataLoader的 "num_workers" 参数置为默认值。

### 参考

课程视频 https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r 不是cs231n，但课程内容高度相似。主讲是讲过cs231n的助教，课程19年录制，比广为流传的17年版cs231n新一些。

代码参考 https://github.com/tc5z/2022_CS231N 码风优良，学到虚脱，偷偷放个链接（逃

#### 摆烂部分

assignment2 layernorm向量实现未通过，后续调用的都是naive版本

assignment3 transformer少量test没过，结果work了但没完全work；附加部分LSTM没做
