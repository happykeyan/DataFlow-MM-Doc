---
title: Audio环境安装
icon: material-symbols-light:download-rounded
createTime: 2025/06/09 10:29:31
permalink: /zh/mm_guide/install_audio_understanding/
---
## 环境安装

```bash
conda create -n myvenv python=3.10
conda activate myvenv

cd ./DataFlow-MM
pip install -e .[audio]
```