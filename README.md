# TIANCHI_ALImusic
TAINCHI_ALImusic
在音乐预测中 是要预测每个歌手的未来播放趋势，我的方法是并不是所有的模型都用一个模型或者规则，
而是先将歌手进行分类，按照设置递减率，最后五个周平均递减率，作为切分点分出这些特殊的歌手，
这些歌手采用时间序列，剩下的歌手根据前一个月，第三个月，最后一个月的均值之差的递减率的均值，
分为上升趋势，下降趋势，这些采用多项式回归；剩下较为平稳的采用模型或者规则。但是为每个模型训练出具体的规则与参数。
代码也只是部分代码
