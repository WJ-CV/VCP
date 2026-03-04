<h2 align="center">
✨ VCP: Visual Consensus Prompting for Co-Salient Object Detection ✨
</h2>


## 📢 News
- **[2026/03/04]** 🚀 Released VCP weights [(DUT_class+COCO-SEG)](#VCP_model) and inference code.
- **[2025/06/17]** 👉 paper [link](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Visual_Consensus_Prompting_for_Co-Salient_Object_Detection_CVPR_2025_paper.pdf)
- **[2025/02/27]** 🎉🎉🎉 Accepted to CVPR 2025.


<p align="center">
  <img src="https://github.com/user-attachments/assets/23663485-1314-4e8c-892d-b8e87430e065" width="45%" style="display:inline; margin-right:10px;" />
  <img src="https://github.com/user-attachments/assets/bb71bf10-6314-465d-9efd-e3aeeac4209b" width="45%" style="display:inline;" />
</p>


Framework Overview
---
![3](https://github.com/user-attachments/assets/79903c7a-c1cd-47b0-b003-8e496f80738d)

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
