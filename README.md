<div align="center">

# STM3: Mixture of Multiscale Mamba for Long-Term Spatio-Temporal Time-Series Prediction (KDD 2026)

<a href="https://arxiv.org/abs/2508.12247"><img src="https://img.shields.io/badge/arXiv-2508.12247-b31b1b.svg?style=for-the-badge" alt="arXiv"></a>
<a><img src="https://img.shields.io/badge/KDD-2026-4B8BBE.svg?style=for-the-badge" alt="KDD 2026"></a>

</div>

STM3 is designed for long-term spatio-temporal forecasting, where multiscale temporal patterns and heterogeneous spatial dependencies are deeply entangled. It addresses this challenge with three core components:

- **Multiscale Mamba** captures long-range temporal dynamics across multiple scales within an efficient Mamba block.
- **AGCCN** performs causal scale-wise spatio-temporal fusion, preserving scale distinguishability while modeling adaptive inter-node correlations.
- **DMoE** routes spatial nodes with static embeddings and contrastive regularization, disentangling heterogeneous node-level patterns across experts.

<p align="center">
  <img src="assets/main.png" alt="STM3 overview" width="95%"/>
</p>

## News

- [2026.05]: 🎉 STM3 has been accepted by **KDD 2026**!

## Installation

We recommend using a Conda environment.

```bash
conda create -n stpredict python=3.10
conda activate stpredict
pip install -r requirements.txt
```

Some dependencies, such as `mamba-ssm` and `causal-conv1d`, are sensitive to CUDA, PyTorch, and compiler versions. If installation fails, install these packages following the version requirements of your local CUDA environment.

## Data

Datasets are available [here](https://zenodo.org/records/17946270).


After downloading, place each dataset in the corresponding subdirectory under `data/`.

## Quick Start

```bash
python run.py --dataset METR_LA --model STM3 --lag 96 --horizon 96 --cuda_devices 0
```

## Citation

If you find this repository useful, please cite:

```bibtex
@article{chen2025stm3,
  title={STM3: Mixture of Multiscale Mamba for Long-Term Spatio-Temporal Time-Series Prediction},
  author={Chen, Haolong and Zhang, Liang and Xin, Zhengyuan and Zhu, Guangxu},
  journal={arXiv preprint arXiv:2508.12247},
  year={2025}
}
```

## Further Reading

1. [An overview of domain-specific foundation model: key technologies, applications and challenges](https://link.springer.com/article/10.1007/s11432-025-4498-2) in *SCIS 2026*: Broader discussion of key technologies, applications, and challenges in **domain-specific foundation models**.

```bibtex
@article{chen2026overview,
  title={An overview of domain-specific foundation model: key technologies, applications and challenges},
  author={Chen, Haolong and Chen, Hanzhi and Zhao, Zijian and Han, Kaifeng and Zhu, Guangxu and Zhao, Yichen and Du, Ying and Xu, Wei and Shi, Qingjiang},
  journal={Science China Information Sciences},
  volume={69},
  number={1},
  pages={111301},
  year={2026},
  publisher={Springer}
}
```

2. [DK-Root: A Joint Data-and-Knowledge-Driven Framework for Root Cause Analysis of QoE Degradations in Mobile Networks](https://arxiv.org/abs/2511.11737), *arXiv 2025*: Root cause **classification of QoE degradations** using multidimensional time-series KPIs **in mobile networks**.

```bibtex
@article{li2025dk,
  title={DK-Root: A Joint Data-and-Knowledge-Driven Framework for Root Cause Analysis of QoE Degradations in Mobile Networks},
  author={Li, Qizhe and Chen, Haolong and Li, Jiansheng and Chai, Shuqi and Li, Xuan and Hou, Yuzhou and Shao, Xinhua and Li, Fangfang and Han, Kaifeng and Zhu, Guangxu},
  journal={arXiv preprint arXiv:2511.11737},
  year={2025}
}
```

3. [FedRMamba: Federated Residual Mamba for Multivariate Time-Series Forecasting](https://dl.acm.org/doi/abs/10.1145/3774904.3792712) in *WWW 2026*: A **federated foundation model for multivariate time-series forecasting** with residual Mamba architectures.

```bibtex
@inproceedings{hu2026fedrmamba,
  title={FedRMamba: Federated Residual Mamba for Multivariate Time-Series Forecasting},
  author={Hu, Zhiwei and Zhang, Liang and Zhu, Guangxu},
  booktitle={Proceedings of the ACM Web Conference 2026},
  pages={7610--7620},
  year={2026}
}
```

4. [CSI-BERT2: A BERT-inspired Framework for Efficient CSI Prediction and Classification in Wireless Communication and Sensing](https://ieeexplore.ieee.org/abstract/document/11278110) in *TMC 2025*: Efficient **prediction and classification** of CSI time-series data in **wireless communication and sensing**.

```bibtex
@article{zhao2025csi,
  title={CSI-BERT2: A BERT-inspired Framework for Efficient CSI Prediction and Classification in Wireless Communication and Sensing},
  author={Zhao, Zijian and Meng, Fanyi and Lyu, Zhonghao and Li, Hang and Li, Xiaoyang and Zhu, Guangxu},
  journal={IEEE Transactions on Mobile Computing},
  year={2025},
  publisher={IEEE}
}
```
