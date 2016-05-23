# Implicit Relationship Detection


## 思路
首先简单介绍一下我的思路：我想使用CNN的方法来进行分类。

### 网络结构


**输入**：两句话

**输出**：分类的类别（是一个one-of-k的向量）


## 程序

### 程序结构

```
utils
|--nn.py 
|  |--CNN
|--reader.py #用于读取数据、建立词表
|  |--Reader #读取数据、标签
|  |--Vocab  #建立词表
```

### 使用

**配置**
```python
{
    "vocab_size": 100000,
    "maxlen": 450,
    "batch_size": 30,
    "embedding_dims": 100,
    "nb_filter": 250,
    "filter_length":3,
    "hidden_size": 300,
    "nb_epoch": 10,
    "dropout": 0.5, 
    "train_file": "data/train_pdtb_imp.json",
    "vocab_file": "data/vocab",
    "test_file": "",
    "valid_file": "data/dev_pdtb_imp.json",
    "vocab_size": 100000,
}
```


## 实验
### 第一版：双Embedding-CNN结构