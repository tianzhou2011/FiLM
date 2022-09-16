# FiLM

FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting

In long-term forecasting, FiLM achieves SOTA, with a **19% relative improvement** on six benchmarks, covering five practical applications: **energy, traffic, economics, weather and disease**.

![plot](./graph/FilM_overall.png)
![plot](./graph/LMU_LMUR_signal_structure.png)
![plot](./graph/FNO_structure.png)


## Main Results
![plot](./graph/FilM_main_result.png)

## Get Started

1. Install Python 3.9, PyTorch 1.11.0.
2. Download data. You can obtain all the six benchmarks from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/) or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the Multivariate/Univariate experiment results by:

```bash
bash ./scripts/ETT_script/FiLM_ETTm2.sh
bash ./scripts/ECL_script/FiLM.sh
bash ./scripts/Exchange_script/FiLM.sh
bash ./scripts/Traffic_script/FiLM.sh
bash ./scripts/Weather_script/FiLM.sh
bash ./scripts/ILI_script/FiLM.sh


bash ./scripts/ETT_script/FiLM_ETTm2_S.sh
bash ./scripts/ECL_script/FiLM_S.sh
bash ./scripts/Exchange_script/FiLM_S.sh
bash ./scripts/Traffic_script/FiLM_S.sh
bash ./scripts/Weather_script/FiLM_S.sh
bash ./scripts/ILI_script/FiLM_S.sh
```



## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

https://github.com/thuml/Autoformer

