# Wireless Localization Model

This repository contains the official implementation of the paper:  
**"AI-driven Wireless Positioning: Fundamentals, Standards, State-of-the-Art, and Challenges"**

It demonstrates example models for wireless positioning using AI techniques.

We appreciate your feedback — it greatly motivates us and helps improve the project!



## 🔧 How to Run

### 📁 Dataset Preparation

1. Download the dataset and place it under the `./dataset` directory.
2. Generate data index files using:

```bash
python randidx.py
```

### 🏋️‍♂️ train

To train the model run:
```bash
python train.py --model Transformer  --Dataset MaMIMO
```


### 🎯 Test

To test model, run:
```bash
python test_singleBS.py --model Transformer  --Dataset MaMIMO
```



##  Citation

If you find this work helpful, please consider citing:
```bash
@article{pan2025ai,
  title={AI-driven Wireless Positioning: Fundamentals, Standards, State-of-the-art, and Challenges},
  author={Pan, Guangjin and Gao, Yuan and Gao, Yilin and Zhong, Zhiyong and Yang, Xiaoyu and Guo, Xinyu and Xu, Shugong},
  journal={arXiv preprint arXiv:2501.14970},
  year={2025}
}
```

Some other paper:
```bash
@article{pan2025large,
  title={Large Wireless Localization Model (LWLM): A Foundation Model for Positioning in 6G Networks},
  author={Pan, Guangjin and Huang, Kaixuan and Chen, Hui and Zhang, Shunqing and H{\"a}ger, Christian and Wymeersch, Henk},
  journal={arXiv preprint arXiv:2505.10134},
  year={2025}
}
```