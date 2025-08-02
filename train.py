# -*- coding: utf-8 -*-
import os
# import wandb
import gc
from tqdm import tqdm
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from utils.config import parse_args
from dataset import load_dataset
from utils.seed import seed_everything
from utils.collate import collate_fn
from utils.lr_schedule import adjust_learning_rate
from utils.ckpt import _save_checkpoint, _reload_best_model
from utils.evaluate import eval_funcs
from model.graph_llm import GraphLLM
import json
import logging


def main(args):
    # 步骤1：基本配置
    seed = args.seed
    # 创建TensorBoard的SummaryWriter
    log_dir = os.path.join('logs', f'train_{args.dataset}_{args.llm_model_name}_seed{seed}')
    writer = SummaryWriter(log_dir=log_dir)

    seed_everything(seed=args.seed)
    logger.info(args)

    # 步骤2：数据集读取
    dataset = load_dataset[args.dataset]()

    train_dataset, val_dataset, test_dataset = dataset.get_train_eval_test_data()

    # 编写collate_fn函数
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True,
                             shuffle=False, collate_fn=collate_fn)

    # 步骤3：模型创建
    model = GraphLLM(args=args)

    # 步骤4：训练
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = model.print_trainable_params()
    logger.info(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, accum_loss = 0., 0.
    
        for step, batch in enumerate(train_loader):
            try:
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
    
                if (step + 1) % args.grad_steps == 0:
                    adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)
    
                optimizer.step()
                epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()
    
                if (step + 1) % args.grad_steps == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar('Lr', lr, epoch * len(train_loader) + step)
                    writer.add_scalar('Accum Loss', accum_loss / args.grad_steps, epoch * len(train_loader) + step)
                    accum_loss = 0.
    
                progress_bar.update(1)
                logger.info(str(progress_bar))
            except Exception as e:
                logger.error(e)
                progress_bar.update(1)
                logger.info(str(progress_bar))
                continue
    
        logger.info(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        writer.add_scalar('Train Loss (Epoch Mean)', epoch_loss / len(train_loader), epoch)
    
        val_loss = 0.
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = model(batch)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            logger.info(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
            writer.add_scalar('Val Loss', val_loss, epoch)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch
    
        logger.info(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')
    
        if epoch - best_epoch >= args.patience:
            logger.info(f'Early stop at epoch {epoch}')
            break
    
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # 步骤5：评估
    model = _reload_best_model(model, args)
    model.eval()
    eval_output = []
    progress_bar_test = tqdm(range(len(test_loader)))
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            output = model.inference(batch)

            length = len(next(iter(output.values())))
            for i in range(length):
                new_dict = {}
                for key in output:
                    new_dict[key] = output[key][i]
                eval_output.append(new_dict)
        progress_bar_test.update(1)
        logger.info(str(progress_bar_test))

    # 步骤6：后处理和计算指标
    output_dir = f'{args.output_dir}/{args.dataset}'
    os.makedirs(output_dir, exist_ok=True)
    # 获取当前日期和时间（包含年月日）
    from datetime import datetime
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d")
    with open(
            f'{output_dir}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}_{formatted_time}_{len(test_loader)}.json',
            'w', encoding='utf-8') as f:
        for need_jsondict in eval_output:
            json.dump(need_jsondict, f, ensure_ascii=False)
            f.write('\n')
    # path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_path_{args.llm_model_path}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
    # acc = eval_funcs[args.dataset](eval_output, path)
    # print(f'Test Acc {acc}')
    # wandb.log({'Test Acc': acc})
    # 关闭writer
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    handlers = [
        logging.FileHandler(f'/code/KGLLM2.0/MyRAG/logs/train_{args.llm_model_name}.log', mode='w'),
        logging.StreamHandler()  # 打印到终端
    ]
    # logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
    logger = logging.getLogger(__name__)
    for handler in handlers:
        logger.addHandler(handler)
    main(args)

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()

