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
| ** #Negatvie ** | 1,181 | 137 | 290 |
| ** #Neutral ** | 673 | 20 | 60 |
| ** #Total ** | 4,904 | 579 | 1,094 |

### Yelp2018

| **Dataset** | Users | Restaurant | Interaction |
|:-------:|:-------:|:-------:|:-------:|
| ** Train ** | - | - | - |
| ** Valid ** | - | - | - |
| ** Test ** | - | - | - |
