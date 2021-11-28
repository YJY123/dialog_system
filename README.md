# dialog_system
### 意图识别模块——fasttext 分类器在测试集的 F1 score 如下：
![fasttext.PNG](https://i.loli.net/2021/11/28/CcIwVsQF5ZtJ3Ay.png)

### 检索模块——ANNS召回（HNSW）
**1. HNSW 构建：**
![HNSW构建.PNG](https://i.loli.net/2021/11/28/PNuDZQCbsotRk1T.png)

**2. HNSW recall@1：**
![HNSW_recall.PNG](https://i.loli.net/2021/11/28/J1nGwkCyviZPV63.png)

**3. HNSW 召回结果：**
![HNSW_结果1.PNG](https://i.loli.net/2021/11/28/nZGKwiraRxIVBAQ.png)
![HNSW_结果2.PNG](https://i.loli.net/2021/11/28/pBSQzAy6Z72qclT.png)

### 检索模块——LGBMClassifier 训练 L2R 模型
**1. 训练过程：**
![L2R_1.PNG](https://i.loli.net/2021/11/28/hvrI5oAB8psm6qV.png)
![L2R_2.PNG](https://i.loli.net/2021/11/28/HVmJyqXGsPlg72U.png)

**2. L2R 模型在测试集上的评估结果：**
![L2R排序模型在测试集上的结果.PNG](https://i.loli.net/2021/11/28/qj8n3iFlZHd9BfD.png)

### 检索模块——最后的整合结果
![最后检索模型的结果.PNG](https://i.loli.net/2021/11/28/S9y5xbQB1zoXtku.png)

### 生成模块——闲聊模型训练过程如下：
https://colab.research.google.com/drive/12iSbDa8454Bc4qlSpPReHC-DDhnhB0ZG
