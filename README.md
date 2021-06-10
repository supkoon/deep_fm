# Deep_FM with movielens- 1m. tensorflow 2


## --reference paper
Guo, H. et al. “DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.” IJCAI (2017).


## --discription
+ dataset : Movielens -1m
+ predict sentiments 0 ~ 3.5 == > class "0".    4.0~5.0 ==> class "1"


## example : FM
```
python DeepFm.py --path "./datasets/" --dataset "movielens" --embedding_size 8 --dropout_rate 0.5 --epochs 10 --batch_size 32 --lr 0.01 --learner "Adam" --layers 400 400 400 --activation 'relu' --patience 10 --test_size 0.1 --out 1

```
