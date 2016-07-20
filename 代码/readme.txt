1.首先用main.py统计出歌手的日播放量
2.用 statsmodels.graphics.api 模块qqplot 对用户分类，符合正太分布的属于一类，不属于的分为一类，
3.对create_week_for_artist.py 为艺人添加是否是周末这一特征
4.create_artist_play_trainANDtest.py 划分训练集测试集
5.train_artist_selectModel_mean_withoutp.py 训练模型，为艺人训练出合适的模型，不同类别的艺人用不同的模型，有用均值，时间序列，随机森林，多项式拟合
6.result_predict_TS_constant_mean.py  对艺人用不同的模型进行预测得出结果