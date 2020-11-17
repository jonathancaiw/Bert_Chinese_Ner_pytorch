import os
import torch
from net.bert_ner import Bert_CRF
from Io.data_loader import create_batch_iter
from train.train import fit
import config.args as args
from util.porgress_util import ProgressBar
from preprocessing.data_processor import produce_data


def start():
    # 优先使用缓存
    if not os.path.exists(args.TRAIN) or not os.path.exists(args.VALID):
        produce_data()

    if os.path.exists(args.TRAIN_CACHE):
        train_iter, num_train_steps = torch.load(args.TRAIN_CACHE)
    else:
        train_iter, num_train_steps = create_batch_iter("train")

    if os.path.exists(args.VALID_CACHE):
        eval_iter = torch.load(args.VALID_CACHE)
    else:
        eval_iter = create_batch_iter("dev")

    epoch_size = num_train_steps * args.train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs

    pbar = ProgressBar(epoch_size=epoch_size, batch_size=args.train_batch_size)

    model = Bert_CRF.from_pretrained(args.bert_model, num_tag=len(args.labels))

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    fit(model=model,
        training_iter=train_iter,
        eval_iter=eval_iter,
        num_epoch=args.num_train_epochs,
        pbar=pbar,
        num_train_steps=num_train_steps,
        verbose=1)
