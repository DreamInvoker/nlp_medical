import argparse
import os

data_dir = './data/'


def get_opt():
    parser = argparse.ArgumentParser()

    # 数据集位置
    parser.add_argument('train_data', type=str, default=os.path.join(data_dir, 'train.txt'))
    parser.add_argument('dev_data', type=str, default=os.path.join(data_dir, 'dev.txt'))
    parser.add_argument('test_data', type=str, default=os.path.join(data_dir, 'test.txt'))

    # 保存模型和加载模型相关
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--model_name', type=str, default='train_model')
    parser.add_argument('--pretrain_model', type=str, default='')

    # 训练相关
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dev_batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--test_epoch', type=int, default=1)
    parser.add_argument('--log_step', type=int, default=50)  # 记录日志的间隔, 以batch为单位
    parser.add_argument('--save_model_freq', type=int, default=1)  # 保存模型间隔，以epoch为单位

    # 模型的一些settings

    return parser.parse_args()
