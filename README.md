# ToDo

1. 搭建TFT生产环境 and Task1
   描述、解释TFT模型；撰写Task1内容（实施细节、预测结果……）

2. Task 2 (Try TFT or build advanced model based on TFT)
   撰写task2内容，描述、解释为Task2做的额外工作
   合稿（包括中期报告、最终报告）

3. Task 3 (Try TFT and other traditional or new methods)
   撰写task3内容，描述、解释为Task3做的额外工作。描述、解释洞察。

4. 撰写任务描述、数据集收集工作（参考project proposal）
   进行数据探索和可视化，并撰写此部分报告
   制作最终PPT

# Project Proposal

## Project problem

​	Civil aviation is an important part of the transportation system, but due to a number of factors including, but not limited to, weather, military control, major events, emergencies, etc., civil flights have a high probability of being delayed for a long period of time, which may last for hours or even longer. This causes a lot of trouble for passengers' travel plans.
​	In order to minimize the disturbance caused by unexpected flight delays, we intend to predict flight delays based on flight data. The detailed problem set is formulated below:

1. Predict flight delays based on all information before planes land on the runway (wheel on time).
   This task is set to be the benchmark task, which should be the easiest one, for most emergencies take place before the wheel on time. Accuracy is the key criteria of this task.
2. Predict flight delays based on all information before planes leave the runway (wheel off time).
   In view of the short time between landing and actual arrival, it is not very meaningful to give a prediction after the aircraft has landed, the second task is to give a prediction of the length of the delay of the aircraft based on the information already available before the aircraft takes off (including the information known to the tower at the time of takeoff, such as military control, major events, etc.).
3. Use interpretable time series modeling to gain insights from segmentation of different airports/departure flights, etc.
   Use an interpretable time series model, forecast data from different airports/flights separately and analyze the differences in the importance of different features in each group, resulting into insights after other necessary analyses.

## Dataset

​	The U.S. Department of Transportation's (DOT) Bureau of Transportation Statistics tracks the on-time performance of domestic flights operated by large air carriers.
[dataset](https://www.kaggle.com/datasets/usdot/flight-delays)

## Methods plan to use

​	We plan to base our method on the paper [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363), in which a interpretable transformer structure temporal model is introduced. This model has build-in interpretability and state-of-art prediction accuracy.


