

python absa_train.py --semeval_dir dataset/SemEval --yelp_dir dataset/Yelp2018 --fix_tfm 0 \
                --max_seq_length 512 --num_epochs 10 --batch_size 128 \
                --save_steps 100 --seed 42 --warmup_steps 0 \
                --model_name_or_path bert-base-uncased \
                --max_grad_norm 1.0 --device cuda
