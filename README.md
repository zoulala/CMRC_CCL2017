
# 中文机器阅读理解-完形填空
中文机器阅读理解（Chinese Machine Reading Comprehension）之完形填空

“讯飞杯”竞赛参考：http://www.hfl-tek.com/cmrc2017/task/

项目参考：https://github.com/bojone/CCL_CMRC2017，主要参考该项目，并做适当梳理调整。

# 基本原理
[参考文档](https://kexue.fm/archives/4564)

填空xxxx的上文L和下文R，分别进入bi_LSTM,得到时间序列向量outputs,和最后状态向量state
> outputs_L, state =bi_LSTM(L)  ；outputs_R, state =bi_LSTM(R)

将outputs_L,outputs_R 连接起来:outputs，state_L+state_R 取平均:state

在用state分别与outputs向量进行内积 ，计算sorfmax概率最大的项，所对应的词即为填空词。
>  index = np.argmax(sorfmax(matul(outputs, state)))

# 数据
https://github.com/ymcui/cmrc2017

# 训练

> python train.py
```
start to training...
step: 2/20000...  loss: 42.0146...  3.4617 sec/batch
step: 4/20000...  loss: 40.0625...  3.2386 sec/batch
step: 6/20000...  loss: 37.7302...  3.0536 sec/batch
```

# 验证精度：






