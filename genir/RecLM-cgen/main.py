import random
import numpy as np
import torch
import transformers
from trainer import SFTTrainer
from train_utils.dataset import SFTDataset, ValueLinkingDataset, Train_task_group_mapping, Val_task_group_mapping, Test_task_group_mapping, ValueLinking_group, ValValueLinking_group
from train_utils.param import Config, get_args


if __name__ == '__main__':
    args = get_args()
    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    transformers.set_seed(args.seed)

    if args.train_stage == 'SFT' and args.share_chat_gpt_ratio > 0.:
        args.SFT_train_tasks = args.SFT_train_tasks + ',ShareChatGPT'

    trainer = SFTTrainer(Config(**vars(args)))
    if args.train_stage == 'SFT' or args.train_stage == 'ValueLinking_SFT':
        trainer.SFT_train()
    elif args.train_stage == 'SFT_Embedding':
        trainer.SFTEmbedding_train()
    elif args.train_stage in ['SFT_Test', 'SFT_Embedding_Test', 'ValueLinking_Test']: # Added a test stage for our task
        if args.SFT_test_task == "SFTTestSeqRec-CS-MR":
            trainer.SFT_Embedding_MR_test()
        else:
            trainer.SFT_test()
    elif args.train_stage == 'SFT_Merge':
        trainer.SFT_adapter_merge()
    else:
        raise NotImplementedError
