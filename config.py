import argparse
import os

import tokenizers

data_dir = './data/'


class PLMConfig:
    MODEL_PATH = './PLM/bert-base-chinese'
    VOCAB_PATH = f'{MODEL_PATH}/vocab.txt'
    tokenizer = tokenizers.BertWordPieceTokenizer(
        vocab_file=f'{VOCAB_PATH}',
        lowercase=True
    )


def get_opt():
    parser = argparse.ArgumentParser()

    # 数据集位置
    parser.add_argument('--train_data', type=str, default=os.path.join(data_dir, 'train.txt'))
    parser.add_argument('--dev_data', type=str, default=os.path.join(data_dir, 'dev.txt'))
    parser.add_argument('--test_data', type=str, default=os.path.join(data_dir, 'test.txt'))

    # 保存模型和加载模型相关
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--model_name', type=str, default='train_model')
    parser.add_argument('--pretrain_model', type=str, default='')

    # 训练相关
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dev_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_warmup_steps', type=int, default=500)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--test_epoch', type=int, default=1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--save_model_freq', type=int, default=1)  # 保存模型间隔，以epoch为单位

    # 模型的一些settings
    parser.add_argument('--dropout', type=float, default=0.5)

    return parser.parse_args()
