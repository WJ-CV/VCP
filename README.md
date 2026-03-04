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


<p align="center">
  <img src="https://github.com/user-attachments/assets/23663485-1314-4e8c-892d-b8e87430e065" width="45%" style="display:inline; margin-right:10px;" />
  <img src="https://github.com/user-attachments/assets/bb71bf10-6314-465d-9efd-e3aeeac4209b" width="45%" style="display:inline;" />
</p>


Framework Overview
---
![3](https://github.com/user-attachments/assets/79903c7a-c1cd-47b0-b003-8e496f80738d)

<a name="VCP_model"></a>
## 🏛️ Model Zoo
| VCP_Model | Segformer | prediction results |
|:-----:|:-------:|:-------:|
| [DUT_class+COCO-SEG](https://huggingface.co/wang-jie825/VCP_model/tree/main) | [b4](https://huggingface.co/wang-jie825/VCP_model/tree/main) | [google-drive](https://drive.google.com/file/d/1roiadSPrNQjylI3cS433GssQ4-lMSBi4/view?usp=sharing) [Hug](https://huggingface.co/datasets/wang-jie825/VCP_CoSOD_result/tree/main) |

Results
---
The prediction results of our model are available on [google-drive](https://drive.google.com/file/d/1roiadSPrNQjylI3cS433GssQ4-lMSBi4/view?usp=sharing)|[BaiduYun](https://pan.baidu.com/s/18UAm2KGET9itUdNI9F8eRw)(fetch code: 0825)

Quantitative and qualitative comparison with SOTA methods
---
![c93a3a49bc1bd12a41683c1be8b3c1b2](https://github.com/user-attachments/assets/b4365436-4f5b-4493-8251-a7ce48dbf64a)
<img width="1051" alt="fig 6" src="https://github.com/user-attachments/assets/a209991e-4707-4a00-9556-36882923f588" />

Some alternative schemes of VCP exhibit more efficient parameters and competitive performance
---
![6](https://github.com/user-attachments/assets/4b0c9130-b028-417f-90e1-ed8584f07f96)

Extention to RGB-D CoSOD task
---
We use the most straightforward early fusion strategy, which does not introduce additional parameters, to validate the effectiveness and generalization of the proposed VCP for the RGB-D CoSOD task. Quantitative and qualitative comparison with SOTA methods：
![36b103e8f61f99d42c0a648cd44a1b1f](https://github.com/user-attachments/assets/3d65892b-6bb7-4487-8eff-23025fa2a2aa)
![8](https://github.com/user-attachments/assets/1149ffd4-f7a6-4dec-8a0c-1c55c8a773dc)

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
