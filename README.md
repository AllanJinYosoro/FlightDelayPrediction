我们的源代码请见：
https://github.com/AllanJinYosoro/FlightDelayPrediction/tree/main

数据：
链接：https://pan.baidu.com/s/1Htpob4AQNRFFtDFVHP7Udw 
提取码：wkoi 
--来自百度网盘超级会员V6的分享

其中名为flights的为原始数据，名为data_preprocessed的为预处理数据，parquet形式更方便python快速读取。

①初始预处理：/data/data processing.ipynb（其中需要的辅助文件均在同一文件夹下，生成的数据即data_preprocessed，可以直接用后者进行后续步骤）
②定期航班与巡回航班的区分：/data/circuit flight/dense_flight.py 与 /data/continuous flight/continuous_flight.py
③ARIMA模型设置：/model/ARIMA.py（可直接在其他文件中调用）
④ARIMA模型搭建与分析：/analysis/circuit.ipynb 与 /analysis/scheduled.ipynb
⑤TFT模型搭建与分析：/model/tft model.ipynb 与 /model/tft load.ipynb（后者可直接加载训练好的模型）