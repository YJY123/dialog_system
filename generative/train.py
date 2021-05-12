#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FilePath: /dialog_system/generative/train.py
Desciption: Train a BERT seq2seq model.
'''
import torch 
from tqdm import tqdm
import torch.nn as nn 
from torch.optim import Adam
import pandas as pd
import numpy as np
import os
import json
import sys
import csv

sys.path.append('..')
cur = '/content/drive/MyDrive/dialog'
sys.path.append(cur)
from config import batch_size, lr, root_path, bert_chinese_model_path, log_path, max_length, max_grad_norm, gradient_accumulation
from generative.seq2seq import Seq2SeqModel
from generative.bert_model import BertConfig
import time
from torch.utils.data import Dataset, DataLoader
from generative.tokenizer import Tokenizer, load_chinese_base_vocab
from utils.tools import create_logger

global logger
logger = create_logger(log_path)


def read_corpus(data_path):
    df = pd.read_csv(data_path,
                     sep='\t',
                     header=None,
                     names=['src', 'tgt'],
                     quoting=csv.QUOTE_NONE
                    ).dropna()
    sents_src = []
    sents_tgt = []
    for index, row in df.iterrows():
        query = row["src"]
        answer = row["tgt"]
        sents_src.append(query)
        sents_tgt.append(answer)
    return sents_src, sents_tgt


# 自定义dataset
class SelfDataset(Dataset):
    """
    针对数据集，定义一个相关的取数据的方式
    """
    def __init__(self, path):
        # 一般init函数是加载所有数据
        super(SelfDataset, self).__init__()
        # 读原始数据
        self.sents_src, self.sents_tgt = read_corpus(path)
        self.word2idx = load_chinese_base_vocab()
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.tokenizer = Tokenizer(self.word2idx)

    def __getitem__(self, i):
        # 得到单个数据

        src = self.sents_src[i] if len(self.sents_src[i]) < max_length else self.sents_src[i][:max_length]
        tgt = self.sents_tgt[i] if len(self.sents_tgt[i]) < max_length else self.sents_tgt[i][:max_length]

        token_ids, token_type_ids = self.tokenizer.encode(src, tgt)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):

        return len(self.sents_src)


def collate_fn(batch):
    """
    动态padding， batch为一部分sample
    """
    def padding(indice, max_length, pad_idx=0):
        """
        pad 函数
        注意 token type id 右侧pad是添加1而不是0，1表示属于句子B
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    # token_type_ids_padded = padding(token_type_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length, 1)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded


class Trainer:
    def __init__(self, checkpoint=False):
        self.checkpoint = checkpoint
        self.pretrain_model_path = bert_chinese_model_path
        self.batch_size = batch_size
        self.lr = lr
        logger.info('加载字典')
        self.word2idx = load_chinese_base_vocab()
        # 判断是否有可用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info('using device:{}'.format(self.device))
        # 定义模型超参数
        bertconfig = BertConfig(vocab_size=len(self.word2idx))
        logger.info('初始化BERT模型')
        self.bert_model = Seq2SeqModel(config=bertconfig)
        logger.info('加载预训练的模型～')
        if not self.checkpoint:
            logger.info('用原始Bert参数加载模型～')
            self.load_model(self.bert_model, self.pretrain_model_path)
        else:
            logger.info('用checkpoint参数加载模型～')
            self.bert_model.load_state_dict(torch.load(os.path.join(root_path, 'model/generative/bert.model.epoch.12')))
        logger.info('将模型发送到计算设备(GPU或CPU)')
        self.bert_model.to(self.device)
        logger.info(' 声明需要优化的参数')
        self.optim_parameters = list(self.bert_model.parameters())
        self.init_optimizer(lr=self.lr)
        # 声明自定义的数据加载器

        logger.info('加载训练数据')
        train = SelfDataset(os.path.join(root_path, 'data/generative/train.tsv'))
        logger.info('加载测试数据')
        dev = SelfDataset(os.path.join(root_path, 'data/generative/dev.tsv'))
        self.trainloader = DataLoader(train, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        self.devloader = DataLoader(dev, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def init_optimizer(self, lr):
        # 用指定的学习率初始化优化器
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=1e-3)

    def load_model(self, model, pretrain_model_path):

        checkpoint = torch.load(pretrain_model_path)
        # 模型刚开始训练的时候, 需要载入预训练的BERT

        checkpoint = {k[5:]: v for k, v in checkpoint.items()
                      if k[:4] == "bert" and "pooler" not in k}
        model.load_state_dict(checkpoint, strict=False)
        torch.cuda.empty_cache()
        logger.info("{} loaded!".format(pretrain_model_path))

    def train(self, epoch):
        # 一个epoch的训练
        logger.info('starting training')
        self.bert_model.train()
        self.iteration(epoch, dataloader=self.trainloader)
        self.evaluate(self.devloader)
        logger.info('training finished')

    def iteration(self, epoch, dataloader):
        total_loss = 0
        batch_time_avg = 0.0
        start_time = time.time() ## 得到当前时间
        # for batch_idx, data in enumerate(tqdm(dataloader, position=0, leave=True)):
        tqdm_batch_iterator = tqdm(dataloader, position=0, leave=True)
        for batch_idx, data in enumerate(tqdm_batch_iterator):
            batch_start = time.time()
            token_ids, token_type_ids, target_ids = data
            token_ids = token_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            # 因为传入了target标签，因此会计算loss并且返回
            enc_layers, logits, loss, attention_layers = self.bert_model(token_ids,
                                                                         token_type_ids,
                                                                         labels=target_ids)
            # loss = loss / gradient_accumulation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bert_model.parameters(), max_grad_norm)
            # if (batch_idx + 1) % gradient_accumulation == 0:
            # 为计算当前epoch的平均loss
            total_loss += loss.item()
            # 更新参数
            self.optimizer.step()
            # 清空梯度信息
            self.optimizer.zero_grad()
            batch_time_avg += time.time() - batch_start
            # torch.nn.utils.clip_grad_norm_(self.bert_model.parameters(), max_grad_norm)
            description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
                .format(batch_time_avg / (batch_idx + 1), total_loss / (batch_idx + 1))
            tqdm_batch_iterator.set_description(description)
        end_time = time.time()
        spend_time = end_time - start_time
        epoch_loss = total_loss / len(dataloader)
        # 打印训练信息
        logger.info(f"epoch is {epoch}. loss is {epoch_loss:06}. spend time is {spend_time}")
        print("-> Training time: {:.4f}s, loss = {:.4f}, epoch = {}".format(spend_time, epoch_loss, epoch))
        # 保存模型
        self.save_state_dict(self.bert_model, epoch)

    def evaluate(self, dataloader):
        logger.info("start evaluating model")
        self.bert_model.eval()
        logger.info('starting evaluating')
        with torch.no_grad():
            # for token_ids, token_type_ids, target_ids in tqdm(self.devloader, position=0, leave=True):
            total_loss = 0
            start_time = time.time()
            for (token_ids, token_type_ids, target_ids) in dataloader:
                token_ids = token_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                enc_layers, logits, loss, attention_layers = self.bert_model(token_ids,
                                                                             token_type_ids,
                                                                             labels=target_ids)

                total_loss += loss.item()
                # loss, accuracy = calculate_loss_and_accuracy(outputs, labels=input_ids, device=device)
                # loss = loss.mean()
                # accuracy = accuracy.mean()
        spend_time = time.time() - start_time
        epoch_loss = total_loss / len(dataloader)
        logger.info("evaluate time {} ,loss {}".format(spend_time, epoch_loss))
        print("-> Valid time: {:.4f}s, loss = {:.4f}".format(spend_time, epoch_loss))
        logger.info("finishing evaluating")

    def save_state_dict(self, model, epoch, file_path="model/generative/bert.model"):
        """存储当前模型参数"""
        save_path = os.path.join(root_path, file_path + ".epoch_new.{}".format(str(epoch)))
        torch.save(model.state_dict(), save_path)
        logger.info("{} saved!".format(save_path))


if __name__ == "__main__":
    trainer = Trainer(checkpoint=True)
    train_epoches = 30
    # best_loss = float('inf')

    for epoch in range(train_epoches):
        # 训练一个epoch
        trainer.train(epoch)
