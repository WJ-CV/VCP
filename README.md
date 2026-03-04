<h2 align="center">
✨ VCP: Visual Consensus Prompting for Co-Salient Object Detection ✨
</h2>

## 📢 News
- **[2026/03/04]** 🚀 Released VCP weights [(DUT_class+COCO-SEG)](#VCP_model) and inference code.
- **[2025/06/17]** 👉 paper [link](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Visual_Consensus_Prompting_for_Co-Salient_Object_Detection_CVPR_2025_paper.pdf)
- **[2025/02/27]** 🎉🎉🎉 Accepted to CVPR 2025.


Abstract
---
Existing co-salient object detection (CoSOD) methods generally employ a three-stage architecture (i.e., encoding, consensus extraction & dispersion, and prediction) along with a typical full fine-tuning paradigm. Although they yield certain benefits, they exhibit two notable limitations: 1) This architecture relies on encoded features to facilitate consensus extraction, but the meticulously extracted consensus does not provide timely guidance to the encoding stage. 2) This paradigm involves globally updating all parameters of the model, which is parameter-inefficient and hinders the effective representation of knowledge within the foundation model for this task. Therefore, in this paper, we propose an interaction-effective and parameter-efficient concise architecture for the CoSOD task, addressing two key limitations. It introduces, for the first time, a parameter-efficient prompt tuning paradigm and seamlessly embeds consensus into the prompts to formulate task-specific Visual Consensus Prompts (VCP). Our VCP aims to induce the frozen foundation model to perform better on CoSOD tasks by formulating task-specific visual consensus prompts with minimized tunable parameters. Concretely, the primary insight of the purposeful Consensus Prompt Generator (CPG) is to enforce limited tunable parameters to focus on co-salient representations and generate consensus prompts. The formulated Consensus Prompt Disperser (CPD) leverages consensus prompts to form task-specific visual consensus prompts, thereby arousing the powerful potential of pre-trained models in addressing CoSOD tasks. Extensive experiments demonstrate that our concise VCP outperforms 13 cutting-edge full fine-tuning models, achieving the new state of the art (with 6.8% improvement in F_m metrics on the most challenging CoCA dataset).

![3](https://github.com/user-attachments/assets/79903c7a-c1cd-47b0-b003-8e496f80738d)

🏗️ Consensus Prompt Generator & Consensus Prompt Disperser
---
<p align="center">
  <img src="https://github.com/user-attachments/assets/3668d236-0dc8-4e99-8b2f-a802a898c6b6" width="45%" style="display:inline; margin-right:10px;" />
  <img src="https://github.com/user-attachments/assets/bb71bf10-6314-465d-9efd-e3aeeac4209b" width="45%" style="display:inline;" />
</p>

<a name="VCP_model"></a>
## 🏛️ Model Zoo
| VCP_Model | Segformer | Prediction results |
|:-----:|:-------:|:-------:|
| [DUT_class+COCO-SEG](https://huggingface.co/wang-jie825/VCP_model/tree/main) | [b4](https://huggingface.co/wang-jie825/VCP_model/tree/main) | [google-drive](https://drive.google.com/file/d/1roiadSPrNQjylI3cS433GssQ4-lMSBi4/view?usp=sharing) [Hug](https://huggingface.co/datasets/wang-jie825/VCP_CoSOD_result/tree/main) |

Quantitative and qualitative comparison with SOTA methods
---
![c93a3a49bc1bd12a41683c1be8b3c1b2](https://github.com/user-attachments/assets/b4365436-4f5b-4493-8251-a7ce48dbf64a)
<img width="1051" alt="fig 6" src="https://github.com/user-attachments/assets/a209991e-4707-4a00-9556-36882923f588" />

Extention to RGB-D CoSOD task
---
We use the most straightforward early fusion strategy, which does not introduce additional parameters, to validate the effectiveness and generalization of the proposed VCP for the RGB-D CoSOD task. Quantitative and qualitative comparison with SOTA methods：
![36b103e8f61f99d42c0a648cd44a1b1f](https://github.com/user-attachments/assets/3d65892b-6bb7-4487-8eff-23025fa2a2aa)
![8](https://github.com/user-attachments/assets/1149ffd4-f7a6-4dec-8a0c-1c55c8a773dc)

🏁 Quick Start
---

```bash
# 1️⃣ Create and activate the conda environment
conda create -n CoSOD python=3.10 -y
conda activate CoSOD

# 2️⃣ Install PyTorch + Torchvision (CUDA 11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113

# 3️⃣ Install OpenMMLab key packages
pip install mmcv-full==1.7.1
pip install mmsegmentation==0.30.0
pip install mmcls==0.25.0

📌 Citation
===
```
@inproceedings{wang2025visual,
  title={Visual consensus prompting for co-salient object detection},
  author={Wang, Jie and Yu, Nana and Zhang, Zihao and Han, Yahong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={9591--9600},
  year={2025}
}
```
