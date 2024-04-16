# Note foe TFT

[arkiv link](https://arxiv.org/abs/1912.09363)

APA style citation: Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, *37*(4), 1748-1764.

## Interpretability Ability

TFT model is able to identify:

1. globally-important variables for the prediction problem
   通过不同特征的Attention weights，分别对Static/Past/Known Future变量提供特征重要性
   <img src="./assets/image-20240409151146921.png" alt="image-20240409151146921" style="zoom:15%;" />
2. persistent temporal patterns
	通过不同时间下的Attention weights，可以看出季节趋势
   <img src="./assets/image-20240409151250374.png" alt="image-20240409151250374" style="zoom:25%;" />
3. significant events
   通过不同时间下的Attention weights，可以看出异常时间段/点
   <img src="./assets/image-20240409151658857.png" alt="image-20240409151658857" style="zoom:25%;" />



## Framework

<img src="./assets/image-20240408203646774.png" alt="image-20240408203646774" style="zoom:33%;" />

![image-20240409141308198](./assets/image-20240409141308198.png)



### GRN

<img src="./assets/image-20240409141250978.png" alt="image-20240409141250978" style="zoom: 50%;" />

GRN：通过门控机制，给予模型对不同输入使用或不使用非线性处理的自主学习空间，从而避免过拟合。

具体而言，这个门控机制是由ELU输出层和GLU模块（图中Gate）实现的：

1. ELU(Exponential Linear Unit)激活函数,在输入较大时近似线性,在输入较小时赋予模型非线性处理能力。

2. GLU模块为：
   $$
   \operatorname{GLU}_\omega(\boldsymbol{\gamma})=\sigma\left(\boldsymbol{W}_{4, \omega} \boldsymbol{\gamma}+\boldsymbol{b}_{4, \omega}\right) \odot\left(\boldsymbol{W}_{5, \omega} \boldsymbol{\gamma}+\boldsymbol{b}_{5, \omega}\right)
   $$
   

​	其中的sigmoid部分起到门控抑制部分，可以通过输出接近0的数字来抑制不需要非线性处理的部分。
​	并在最终的Add&Norm层施加：
$$
\operatorname{LayerNorm}\left(\boldsymbol{a}+\operatorname{GLU}_\omega\left(\boldsymbol{\eta}_1\right)\right)
$$
​	若GLU输出解决0，则GRN等价于线性变化，近似无效。否则GLU可提供足够的非线性处理。

### Variable Selection Module

<img src="./assets/image-20240409142520679.png" alt="image-20240409142520679" style="zoom:50%;" />

​	为了剔除无关变量，并提供insights，TFT为Static/Past/Known Future变量分别独立引入了Variable Selection Module。
​	其中每个特征在嵌入编码后，经过独立的GRN模块处理；且每个特征在所有时间下的值被flatten后通过GRN，经由Softmax层输出为特征权重。二者相乘以作为特征选择后的特征。

### Static Covariate Encoder

<img src="./assets/image-20240409143323021.png" alt="image-20240409143323021" style="zoom:50%;" />

​	其它时间序列模型在处理静态变量时，只是简单地将其在每个时间步下复制，TFT则编写独立的编码模块。静态变量通过独立GRN处理，产生四个独立的静态变量编码向量，分别参与Variable Selection，LSTM Encoder, Static Enrichment中。

### Masked Interpretable Multi-Head Attention

​	将不同Attention Head结果平均聚合，以此表示不同特征的重要性

# 报告示例

TFT是一种用于多horizon时间序列预测任务的注意力模型。它的设计目标是在实现出色的预测性能的同时,还能够提供对模型预测决策的可解释性解读。TFT的创新之处在于,它专门设计了多个模块来处理通常时间序列预测任务中同时存在的多种数据输入形式:静态特征(static covariates)、过去观测值(past inputs)以及已知的未来信息(known future inputs)。

TFT模型的主要组成模块:

1. 门控残差网络(Gated Residual Network, GRN)
   - 通过门控机制,自适应地调节每一层的非线性程度,避免过度拟合
2. 特征选择网络(Variable Selection Networks)
   - 对静态特征、过去特征和未来特征分别做特征选择和加权,剔除无关变量
3. 静态特征编码器(Static Covariate Encoder)
   - 将静态特征编码为多个向量,融入后续的时间建模中
4. 可解释多头注意力(Interpretable Multi-Head Attention)
   - 修改自Transformer中的多头注意力,使注意力权重可解释重要特征
5. 时序融合解码器(Temporal Fusion Decoder)
   - 融合LSTM编码器和可解释注意力层,捕获近期和远期时序模式
6. 分位数输出(Quantile Outputs)
   - 输出多个分位数,而不仅仅是点估计,以提供预测区间

TFT的三大可解释性能力:

1. 分析全局重要特征(Analyzing Global Variable Importance)

- 通过各个特征在特征选择模块中的注意力权重分布,判断出全局重要特征

1. 可视化持久时序模式(Visualizing Persistent Temporal Patterns)

- 通过注意力层的注意力权重分布,检测出数据中的周期性/季节性等模式

1. 识别重要事件与状态转换(Identifying Significant Events & Regimes)

- 计算每个时间步注意力权重与平均模式的偏离程度,从而发现异常状态

通过这三种方式,TFT不仅能建模复杂的时间序列预测问题,还能为预测决策提供全局性和局部性的解释,提高模型的可解释性和可信度。总的来说,TFT实现了高性能预测与可解释性之间的有效权衡。
