
from __future__ import print_function

import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset

import numpy as np


def create_data(min_num, max_num, n_samples):
    # np.random.randint(1,10,size=5) -> [2,3,8,5,1]
    a_vals = np.random.randint(min_num, max_num, size=n_samples)
    b_vals = np.random.randint(min_num, max_num, size=n_samples)
    fa_oparations = np.random.randint(4, size=n_samples)
    train_data = []
    label_data = []
    for (a_val, b_val, fa_oparation ) in zip(a_vals, b_vals, fa_oparations):
        if fa_oparation==0:
            ans = a_val + b_val
        elif fa_oparation==1:
            ans = a_val - b_val
        elif fa_oparation==2:
            ans = a_val * b_val
        elif fa_oparation==3:
            ans = a_val / b_val
        train_data.append([a_val, b_val, ans])
        label_data.append(fa_oparation)

    # 学習では、入力はnumpyのfloat32、出力はint32型が基本。
    train_data = np.array(train_data, dtype=np.float32)
    label_data = np.array(label_data, dtype=np.int32)

    return train_data, label_data

# chainerで学習するために、chainer.datasets.tuple_dataset型にする
def create_dataset(min_num, max_num, n_samples):
    train_data, label_data = create_data(min_num, max_num, n_samples)
    threshold = len(train_data)//10*9
    # 90%を学習データ、10%をテストデータにする。
    train = tuple_dataset.TupleDataset(train_data[0:threshold], label_data[0:threshold])
    test  = tuple_dataset.TupleDataset(train_data[threshold:],  label_data[threshold:])
    return train, test


# ネットワークを定義する。
class MLP(chainer.Chain):

    # ネットワークで使用するレイヤーを定義
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # Linearは全結合層を意味する。
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, n_out)

    # ネットワークの流れの定義
    def __call__(self, x):
        # reluは活性化関数。
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# 標準出力の色を変えるためだけ。
class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=20,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # モデルを定義
    # Classifierクラスは分類問題を簡単にしてくれるクラス
    model = L.Classifier(MLP(n_units=args.unit, n_out=4))

    # GPU使う場合
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        # GPU使う場合は、numpyではなくcupyを使う。
        import cupy
        xp = cupy
    else:
        xp = np

    # 最適化手法をセットアップ
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # データセットを用意
    train, test = create_dataset(min_num=1, max_num=30, n_samples=3600)

    # chainerでは、明示的にfor文を回して、バックプロパゲーションを行って学習することもできるが
    # trainerクラスを使う事で、難しいことは意識せずに学習できる。

    # 学習・テストデータtrainerクラスに渡すために、iteratorに変換する。
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Trainerを定義
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # 学習の状況をテストデータを使って評価する。
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # 評価したものをログデータにする。
    trainer.extend(extensions.LogReport())

    # ログデータを画像で保存する。
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # ログデータから、標準出力するものを指定する。
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # プログレスバー
    trainer.extend(extensions.ProgressBar())

    # 学習開始
    trainer.run()

    # -----学習終わり-----

    # 全く異なる値；学習していない、30~50の数を使って予測してみる。
    test_datas, test_labels = create_data(min_num=30, max_num=50, n_samples=50)
    afo = {0:'+', 1:'-', 2:'x', 3:'/'} # map afo[0]->'+', afo[2]->'x'
    # 学習済みモデルで予測する。
    preds = model.predictor(xp.array(test_datas))
    # model.predictorで予測した結果は、preds.dataに格納される。
    for (test_data, test_label, pred_label) in zip(test_datas, test_labels, preds.data):
        print('Q: ', test_data[0], ' ? ', test_data[1], ' = ', test_data[2])
        print('T: ? = ', afo[test_label])
        print('P: ? = ', afo[int(np.argmax(pred_label))])
        if np.argmax(pred_label) == test_label:
            print(colors.ok + 'correct!' + colors.close)
        else:
            print(colors.fail + 'uncorrect...' + colors.close)
        print()


if __name__ == '__main__':
    main()
