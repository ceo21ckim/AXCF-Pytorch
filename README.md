# AXCF-Pytorch

Aspect-aware eXplainable Collaborative Filtering

## Requirements

- torch==1.10.0
- transformers==4.24.0
- tensorflow-gpu==2.6.0
- pandas==1.3.5
- numpy==1.19.5

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

## Execute

```
python absa_train.py --semeval_dir dataset/SemEval --yelp_dir dataset/Yelp2018 --fix_tfm 0 \
                --max_seq_length 512 --num_epochs 10 --batch_size 128 \
                --save_steps 100 --seed 42 --warmup_steps 0 \
                --model_name_or_path bert-base-uncased \
                --max_grad_norm 1.0 --device cuda
```
