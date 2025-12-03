# SDPDRec: Synergistic Dual-path with Personalized Diffusion for Multimodal Recommendation

Official implementation of our paper:  
**SDPDRec: Synergistic Dual-path with Personalized Diffusion for Multimodal Recommendation**

> Authors: Yongfu Zha, Wei Yu, Yufei Wang, Jie Peng, Cui Miao, Xinxin Dong, Xiaodong Wang*  
> National University of Defense Technology, China  
> *Corresponding author: xdwang@nudt.edu.cn*

---

## 📥 Data Download

The preprocessed datasets (**Baby**, **Sports**, **Clothing**) are provided by [FREEDOM](https://github.com/enoche/FREEDOM).  
👉 Download from: [Google Drive](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing)

> The data already contains:
> - Text features (from Sentence-Transformers)
> - Visual features (from CNN)

---
## 📂 Folder Structure

After downloading, place the dataset folders into the `data/` directory:
SDPDRec/
├── data/
│   ├── Baby/
│   ├── Sports/
│   └── Clothing/
├── main.py
└── README.md
---

## 🚀 How to Run

1. Open this project in **PyCharm**
2. Make sure your Python environment has `torch`, `numpy`, etc.
3. Simply run **`main.py`**

> 💡 By default, the script will train on the **Baby** dataset.  
> To change dataset, modify the argument in `main.py` or pass via command line:

```bash
python main.py --dataset Sports