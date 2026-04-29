import os
import torch
import torch.distributed as dist

from tqdm import tqdm
from models.distillation_model import DistillationLoss, preprocess_distillation_batch

from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertModel
from bert_score import score
import numpy

def multioutput_f1_score(y_true, y_pred):
    if y_true.ndim == 2 and y_pred.ndim == 2:
        f1_scores = [f1_score(y_true[:, i], y_pred[:, i], average='macro') for i in range(y_true.shape[1])]
        return numpy.mean(f1_scores)
    else:
        return f1_score(y_true, y_pred, average='macro')
    
def evaluation_teacher(model, train_config, distil_config, eval_dataloader, steps_per_eval, local_rank):


    eval_loss = 0.0
    eval_f1, eval_bert_score_f1 = 0.0, 0.0
    pbar = tqdm(colour="green", desc="Evaluating Teacher", total=steps_per_eval, dynamic_ncols=True)

    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if isinstance(batch, tuple):
                batch = preprocess_distillation_batch(batch)  
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank if train_config.enable_fsdp or distil_config.enable_fsdp else 'cuda:0')

            outputs = model(**batch)
            teacher_output = outputs[1] if isinstance(outputs, tuple) else outputs
            loss = teacher_output.loss
            eval_loss += loss.detach().float()

            predictions = teacher_output.logits.argmax(-1)
            labels = batch['teacher_labels'] if 'teacher_labels' in batch else batch['labels']
            f1 = multioutput_f1_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_f1 += f1

            pbar.update()



        eval_loss /= steps_per_eval
        eval_f1 /= steps_per_eval
        eval_bert_score_f1 /= steps_per_eval

    return eval_loss, eval_f1, eval_bert_score_f1

def evaluation(epoch, model, train_config, distil_config, eval_dataloader, steps_per_eval, local_rank):
    if train_config.enable_fsdp or distil_config.enable_fsdp: world_size = int(os.environ["WORLD_SIZE"])
    if train_config.distillation: distillation_loss = DistillationLoss(skip_student_eos=True)
    eval_loss = 0.0
    eval_cross_loss = 0.0
    eval_dist_loss = 0.0
    pbar = tqdm(colour="green", desc="Evaluating", total=steps_per_eval, dynamic_ncols=True)
    with torch.no_grad():
        for _, batch in enumerate(eval_dataloader):
            if train_config.distillation:
                batch = preprocess_distillation_batch(batch)
            for key in batch.keys():
                if train_config.enable_fsdp or distil_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')

            if train_config.distillation:
                outputs, teacher_output = model(**batch)
                loss, cross_loss, dist_loss, *_components = distillation_loss(epoch, outputs, teacher_output, batch['student_labels'], batch['teacher_labels'])
                eval_cross_loss += cross_loss.detach().float()
                eval_dist_loss += dist_loss.detach().float()
            else:
                outputs = model(**batch)
                loss = outputs.loss
            eval_loss += loss.detach().float()
            pbar.update()

    if torch.cuda.device_count() > 1 and train_config.enable_fsdp or distil_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        if train_config.distillation:
            dist.all_reduce(eval_cross_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(eval_dist_loss, op=dist.ReduceOp.SUM)

    eval_loss /= steps_per_eval
    eval_cross_loss /= steps_per_eval
    eval_dist_loss /= steps_per_eval
    if train_config.enable_fsdp or distil_config.enable_fsdp:
        eval_loss /= world_size
        eval_cross_loss /= world_size
        eval_dist_loss /= world_size
    eval_ppl = torch.exp(eval_cross_loss if train_config.distillation else eval_loss)
    
    return eval_ppl, eval_loss, eval_cross_loss, eval_dist_loss