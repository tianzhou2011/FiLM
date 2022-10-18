# FiLM

FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting
https://arxiv.org/abs/2205.08897

In long-term forecasting, FiLM achieves SOTA, with a **19% relative improvement** on six benchmarks, covering five practical applications: **energy, traffic, economics, weather and disease**.



|![Figure1](https://raw.githubusercontent.com/tianzhou2011/FiLM/main/graphs/FilM_overall.png)|
|:--:| 
| *Figure 1. Overall structure of FiLM* |

|![image](https://raw.githubusercontent.com/tianzhou2011/FiLM/main/graphs/FNO_structure.png) | ![image](https://raw.githubusercontent.com/tianzhou2011/FiLM/main/graphs/LMU_LMUR_signal_structure.png)
|:--:|:--:|
| *Figure 2. Frequency Enhanced Layer (FEL)* | *Figure 3. Legendre Projection Unit (LPU)* |


## Main Results
![image](https://raw.githubusercontent.com/tianzhou2011/FiLM/main/graphs/FilM_main_result.png)


## Get Started

1. Install Python 3.9, PyTorch 1.11.0.
2. Download data. You can obtain all the six benchmarks from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/) or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the Multivariate/Univariate experiment results by:

```bash
bash ./script/ETT_script/FiLM/FiLM_ETTm2.sh
bash ./script/ECL_script/FiLM/FiLM.sh
bash ./script/Exchange_script/FiLM/FiLM.sh
bash ./script/Traffic_script/FiLM/FiLM.sh
bash ./script/Weather_script/FiLM/FiLM.sh
bash ./script/ILI_script/FiLM/FiLM.sh


bash ./script/ETT_script/FiLM/FiLM_ETTm2_S.sh
bash ./script/ECL_script/FiLM/FiLM_S.sh
bash ./script/Exchange_script/FiLM/FiLM_S.sh
bash ./script/Traffic_script/FiLM/FiLM_S.sh
bash ./script/Weather_script/FiLM/FiLM_S.sh
bash ./script/ILI_script/FiLM/FiLM_S.sh
```



## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

https://github.com/thuml/Autoformer

