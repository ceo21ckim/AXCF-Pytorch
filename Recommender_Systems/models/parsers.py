import argparse
from __init__ import * 

bert_parser = argparse.ArgumentParser(description='setting bert parameter for training and evaluating..')

bert_parser.add_argument('--max_seq_length', type=int, default=128)
bert_parser.add_argument('--dr_rate', type=float, default=0.3)
bert_parser.add_argument('--hidden_dim', type=int, default=312, help='bert hidden dimension.')
bert_parser.add_argument('--model_name_or_path', type=str, default='huawei-noah/TinyBERT_General_4L_312D')
bert_parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate for training')
bert_parser.add_argument('--save_parameters', type=str, default=SAVE_PATH)
bert_parser.add_argument('--batch_size', type=int, default=32)
bert_parser.add_argument('--num_epochs', type=int, default=100)
bert_parser.add_argument('--device', type=str, default='cuda')

bert_args = bert_parser.parse_args()



lstm_parser = argparse.ArgumentParser(decsription='setting lstm parameter for training and evaluating..')

lstm_parser.add_argument('--max_seq_length', type=int, default=128)
lstm_parser.add_argument('--dr_rate', type=float, default=0)
lstm_parser.add_argument('--hidden_dim', type=int, default=64, help='lstm hidden_dimension')
lstm_parser.add_argument('--num_layers', type=int, default=2)
lstm_parser.add_argument('--bidirectional', action='store_true')
lstm_parser.add_argument('--learning_rate', type=float, default=1e-3)
lstm_parser.add_argument('--save_parameters', type=str, default=SAVE_PATH)
lstm_parser.add_argument('--batch_size', type=int, default=2048)
lstm_parser.add_argument('--num_epochs', type=int, default=100)

lstm_args = lstm_parser.parse_args()