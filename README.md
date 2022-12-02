# AXCF-Pytorch

Aspect-aware eXplainable Collaborative Filtering

## Requirements

- torch==1.10.0
- transformers==4.24.0
- tensorflow-gpu==2.6.0
- pandas==1.3.5
- numpy==1.19.5

## Experiment setting

- NVIDIA GeForce RTX 3080 Ti 12GB
- i9-11900F 127RAM
- cuda 11.3 version

## Datasets

### SemEval

SemEval is a series of international natural language processing (NLP) research workshops whose mission is to advance the current state of the art in semantic analysis and to help create high-quality annotated datasets in a range of increasingly challenging problems in natural language semantics. Each year's workshop features a collection of shared tasks in which computational semantic analysis systems designed by different teams are presented and compared.

In this paper, use SemEval2014, SemEval2015, and SemEval2016. 

| **Dataset** | **Train** | **Valid** | **Test** |
|--------:|:--------:|:--------:|:--------:|
| **Sentence #Total** | 5,959 | 851 | 1,703 |
| **Aspect #Positive** | 3,050 | 422 | 744 |
| **#Negatvie** | 1,181 | 137 | 290 |
| **#Neutral** | 673 | 20 | 60 |
| **#Total** | 4,904 | 579 | 1,094 |

### Yelp2018

| **Dataset** | Users | Restaurant | Interaction |
|:-------:|:-------:|:-------:|:-------:|
| **Train** | - | - | - |
| **Valid** | - | - | - |
| **Test** | - | - | - |


## Docker 
**1.clone this repository**
``` 
git clone https://github.com/ceo21ckim/AXCF-Pytorch.git
cd AXCF-Pytorch
```

**2.build Dockerfile**
```
docker build --tag [filename]:1.0 .
```

**3.execute**

```
# Docker version 2.0 or later.
docker run --itd --runtime=nvidia --name axcf -p 8888:8888 -v C:\Users\Name\:/workspace AXCF:1.0 /bin/bash
```

```
# Docker-ce 19.03 or later
docker run -itd --gpus all --name axcf -p 8888:8888 -v C:\Users\Name\:/workspace AXCF:1.0 /bin/bash
```

**4.use jupyter notebook**
```
docker exec -it axcf bash

jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

## Execute

```
python absa_train.py --semeval_dir dataset/SemEval --yelp_dir dataset/Yelp2018 --fix_tfm 0 \
                --max_seq_length 512 --num_epochs 10 --batch_size 128 \
                --save_steps 100 --seed 42 --warmup_steps 0 \
                --model_name_or_path bert-base-uncased \
                --max_grad_norm 1.0 --device cuda
```
