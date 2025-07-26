# LWLM: Large Wireless Localization Model

This repository contains the implementation of the paper:  **"AI-driven wireless positioning: Fundamentals, standards, state-of-the-art, and challenges"**  

It is an example for AI-driven positioning.

Thank you very much for your feedback, it can effectively motivate me and help improve this project.



![Pre-training framework](image.png)


## 🔧 How to Run

### 🏋️‍♂️ Dataset
Download dataset and put it in ./dataset

Run for data index generation
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