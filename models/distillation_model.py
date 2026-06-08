import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import numpy as np
from transformers import AutoTokenizer
import re



def preprocess_distillation_batch(batch):
    batch_dict = {"student_" + key: value for key, value in batch[0].items()}
    batch_dict.update({"teacher_" + key: value for key,
                      value in batch[1].items()})
    return batch_dict

def improved_sort(value):
    sums = value.sum(dim=(0, 1))
    sorted_indices = torch.argsort(sums, descending=True)
    sorted_values = value[:, :, sorted_indices]
    return sorted_values

def normalize(value):
    means = value.mean(dim=-1, keepdim=True)
    stds = value.std(dim=-1, keepdim=True)
    z_score_normalized_student = (value)/ (stds+0.0001)
    return z_score_normalized_student

def KL_wo(y_s, y_t, T=1, position_weights=None):
    """KL/SL-style cross-entropy term used as the SL component of MultiLevelOT.

    Inputs ``y_s``, ``y_t`` are shape ``(B, T, V)``. The vanilla
    formulation averages ``-sum(t * log s)`` over both batch and token
    positions. When ``position_weights`` (shape ``(B, T)``) is provided,
    each position's contribution is multiplied by its weight before the
    mean — this implements Eq. 15 of SpanOT-KD (the SL component
    summed over positions then weighted per Eq. 14). ``None`` recovers
    the original behaviour exactly.
    """
    p_s = F.log_softmax(y_s/T, dim=-1)
    p_t = F.softmax(y_t/T, dim=-1)
    per_pos = -torch.sum(p_t * p_s, dim=-1)  # (B, T)
    if position_weights is not None:
        per_pos = per_pos * position_weights.to(per_pos.dtype)
    return per_pos.mean()

class Sinkhorn_seq(nn.Module):
    """Sequence-level Sinkhorn OT distance between two probability tensors.

    Notation used in the diagnostics:
      * ``epsilon`` is the entropic-regularisation strength in ``K = exp(-C/epsilon)``.
        The paper calls this lambda. Default 0.1.
      * The per-sample Sinkhorn output is scaled by ``0.001`` inside :meth:`forward`
        *and* then by ``0.1`` outside, in :class:`DistillationLoss`. The user-facing
        "gamma" weight is the outer 0.1.
    """

    def __init__(self, T=2):
        super(Sinkhorn_seq, self).__init__()
        self.T = 2
    def sinkhorn_normalized(self,x, n_iters=20):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_loss(self, x, y, epsilon=0.1, n_iters=10, return_cost=False, row_weights=None):
        """Sinkhorn OT loss with optional row-wise weighting (SpanOT-KD).

        ``row_weights`` (Tensor of shape ``(T,)`` or None) implements the
        position-pair weight matrix ``W_ij = omega(i)`` from Eq. 16 of the
        SpanOT-KD methodology: the weight is applied **only after** the
        Sinkhorn normalisation, so the transport plan ``P`` is the same as
        in vanilla MultiLevelOT and only the loss aggregation changes.
        Passing ``None`` (or all-ones) yields byte-for-byte identical
        numerics to the original implementation.
        """
        x = x.float()
        y = y.float()
        Wxy = torch.cdist(x, y, p=1)
        K = torch.exp(-Wxy / epsilon)
        P = self.sinkhorn_normalized(K, n_iters)
        product = P * Wxy
        if row_weights is not None:
            # Eq. 16: W_ij = omega(i) — row weight broadcast over columns.
            # Float-cast guards against fp16 underflow of small mu*delta values.
            product = product * row_weights.to(product.dtype).unsqueeze(1)
        loss = torch.sum(product)
        if return_cost:
            return loss, Wxy.detach()
        return loss

    def forward(self, y_s, y_t, row_weights=None):
        """Sequence-level Sinkhorn loss summed over the batch.

        ``row_weights`` (Tensor of shape ``(B, T)`` or None) is the
        per-sample, per-(teacher-)position weight from SpanOT-KD. It is
        forwarded to :meth:`sinkhorn_loss` for each batch index. ``None``
        recovers the original behaviour exactly.
        """
        softmax = nn.Softmax(dim=-1)
        p_s = softmax(y_s/self.T)
        p_t = softmax(y_t/self.T)
        emd_loss = 0
        for i in range(p_s.shape[0]):
            rw = row_weights[i] if row_weights is not None else None
            emd_loss += 0.001*self.sinkhorn_loss(x=p_s[i], y=p_t[i], row_weights=rw)
        return emd_loss

    def forward_with_diagnostics(self, y_s, y_t, epsilon=0.1, row_weights=None):
        """Same numerical output as :meth:`forward`, plus per-batch Sinkhorn diagnostics.

        Returns ``(emd_loss, diag)`` where ``diag`` is a dict with:
          * ``sinkhorn_per_sample_raw_mean``: mean over the batch of the *unscaled*
            Sinkhorn distance ``sum(P * C)`` (i.e. before the internal ``0.001`` and
            before any external weighting). This is the purest "how far apart are
            the two token distributions under the OT geometry" number.
          * ``sinkhorn_batch_sum_scaled_001``: same as :meth:`forward` output,
            i.e. ``sum_i 0.001 * sinkhorn_loss_i``. Kept for cross-checking the
            returned loss tensor.
          * ``sinkhorn_epsilon``: the ``epsilon`` used in ``K = exp(-C/epsilon)``.
          * ``cost_mean`` / ``cost_max`` / ``cost_min``: statistics of the pairwise
            L1 cost matrix ``C`` pooled across all batch items. Useful for
            confirming whether ``C/epsilon`` is in the regime where Sinkhorn is
            informative: if ``C`` values are << epsilon, ``K`` becomes nearly
            uniform and the transport plan degenerates.
        """
        softmax = nn.Softmax(dim=-1)
        p_s = softmax(y_s / self.T)
        p_t = softmax(y_t / self.T)
        emd_loss = 0
        per_sample_raw = []
        cost_means, cost_maxes, cost_mins = [], [], []
        for i in range(p_s.shape[0]):
            rw = row_weights[i] if row_weights is not None else None
            loss_i, C_i = self.sinkhorn_loss(
                x=p_s[i], y=p_t[i], epsilon=epsilon, return_cost=True,
                row_weights=rw,
            )
            emd_loss = emd_loss + 0.001 * loss_i
            per_sample_raw.append(loss_i.detach())
            cost_means.append(C_i.mean().item())
            cost_maxes.append(C_i.max().item())
            cost_mins.append(C_i.min().item())

        if per_sample_raw:
            raw_mean = float(torch.stack(per_sample_raw).mean().item())
            batch_sum_scaled = float((0.001 * torch.stack(per_sample_raw)).sum().item())
        else:
            raw_mean = 0.0
            batch_sum_scaled = 0.0

        diag = {
            "sinkhorn_per_sample_raw_mean": raw_mean,
            "sinkhorn_batch_sum_scaled_001": batch_sum_scaled,
            "sinkhorn_epsilon": float(epsilon),
            "cost_mean": float(sum(cost_means) / max(1, len(cost_means))),
            "cost_max": float(max(cost_maxes)) if cost_maxes else 0.0,
            "cost_min": float(min(cost_mins)) if cost_mins else 0.0,
        }
        return emd_loss, diag

def greedy_algorithm_adjust_s(t, s):
    batch_size, T, k = t.shape
    _, n, _ = s.shape
    
    # Initialize the adjusted source tensor
    s_adjusted = torch.zeros_like(t)
    
    for b in range(batch_size):
        # Initialize set of available source indices for each batch
        available_indices = list(range(n))
        
        for i in range(T):
            C_min = float('inf')
            j_star = -1
            
            for j in available_indices:
                # Compute cost as the sum of absolute differences for each batch
                C = torch.sum(torch.abs(t[b,:,i] - s[b,:,j]))
                
                if C < C_min:
                    C_min = C
                    j_star = j
            
            # Assign the best matching source vector to the adjusted tensor
            s_adjusted[b,:,i] = s[b,:,j_star]
            
            # Remove the selected index from available indices
            available_indices.remove(j_star)

    return s_adjusted

class DistillationModel(nn.Module):
    def __init__(self, student, teacher, teacher_tokenizer, student_tokenizer):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.eval()

    def forward(self, student_input_ids, student_attention_mask, student_labels, teacher_input_ids, teacher_attention_mask, teacher_labels):
        with torch.no_grad():
            teacher_output = self.teacher(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                labels=teacher_labels
            )

        student_output = self.student(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            labels=student_labels
        )
        return student_output, teacher_output


# change stu label to teacher generation

class DistillationModel2(nn.Module):
    def __init__(self, student, teacher, teacher_tokenizer, student_tokenizer, ignore_index=-100):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.teacher.eval()
        self.ignore_index = ignore_index

    def forward(self, student_input_ids, student_attention_mask, student_labels, teacher_input_ids, teacher_attention_mask, teacher_labels):
        with torch.no_grad():
            teacher_output = self.teacher(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                labels=teacher_labels
            )
            teacher_logits = teacher_output.logits
            teacher_token_ids = torch.argmax(teacher_logits, dim=-1)

        student_answer_index, student_answer_size = self.__get_start_and_size_answers(student_labels)
        teacher_answer_index, teacher_answer_size = self.__get_start_and_size_answers(teacher_labels)


        teacher_answers_text = []
        for i in range(len(teacher_answer_index)):
            start_idx = teacher_answer_index[i]
            end_idx = start_idx + teacher_answer_size[i]
            answer_ids = teacher_token_ids[i, start_idx:end_idx] 
            answer_text = self.teacher_tokenizer.decode(answer_ids)  
            teacher_answers_text.append(answer_text)

        new_student_labels = torch.full_like(student_labels, fill_value=-100) 
        for i, answer_text in enumerate(teacher_answers_text):
            student_start_idx = student_answer_index[i]
            encoded_answer = self.student_tokenizer.encode(answer_text, add_special_tokens=True)  
            end_idx = min(student_start_idx + len(encoded_answer), student_labels.size(1))  
            new_student_labels[i, student_start_idx:end_idx] = torch.tensor(encoded_answer[:end_idx-student_start_idx], dtype=torch.long)

        #print(teacher_answers_text)
        #print(student_labels)
        #print(new_student_labels)


        

        student_output = self.student(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            labels=new_student_labels
        )
        return student_output, teacher_output
    
    def __get_start_and_size_answers(self, answer_tensors):
        answers_index = []
        answers_size = []

        for answer in answer_tensors:
            is_value = answer.eq(self.ignore_index)
            answers_size.append(len(answer) - int(is_value.sum()))
            indices = is_value.nonzero(as_tuple=True)[0]
            if len(indices) == 0 or indices[0] != 0:
                answers_index.append(0)
            else:
                diff_indices = indices[1:] - indices[:-1]
                break_index = (diff_indices != 1).nonzero()
                length = (break_index[0].item() +
                          1) if len(break_index) > 0 else len(indices)
                answers_index.append(length-1)
        return answers_index, answers_size


class DistillationLoss(nn.Module):
    def __init__(self, batch_limit=100, store_path='teacher_logits_partial.npy', crossentropy_weight=1, distillation_weight=1, student_temperature=1, teacher_temperature=1, skip_student_eos=False, skip_teacher_eos=False, ignore_index=-100, debug=False, debug_rank=0, tokenizer_student=None, tokenizer_teacher=None, f=1,
                 span_kd_enabled=False, span_aggregation="mean", span_top_r=0.5, span_low_delta=0.1):
        super().__init__()
        self.crossentropy_weight = crossentropy_weight
        self.distillation_weight = distillation_weight
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.skip_student_eos = skip_student_eos
        self.skip_teacher_eos = skip_teacher_eos
        self.ignore_index = ignore_index
        self.debug_rank = debug_rank
        self.debug = debug
        self.f = f

        # SpanOT-KD configuration. When `span_kd_enabled` is False, the
        # forward pass is byte-for-byte identical to the original
        # MultiLevelOT objective. When True, we compute a per-sample,
        # per-position weight tensor (see train.span_ot) and multiply it
        # into the HAD, SL, and SD components per Eqs. 15-17 of the
        # SpanOT-KD methodology.
        self.span_kd_enabled = span_kd_enabled
        self.span_aggregation = span_aggregation
        self.span_top_r = float(span_top_r)
        self.span_low_delta = float(span_low_delta)
        # Tokenisers are required to identify aligned spans (character-level
        # alignment between the student and teacher tokenisations of the
        # ground-truth answer). Loaded eagerly here so the SpanOT path does
        # not pay a per-step from_pretrained cost. They are also loaded
        # under `debug=True` for the existing diagnostic prints, so we
        # consolidate both code paths into one `if needed` branch.
        self.student_tokenizer = None
        self.teacher_tokenizer = None
        need_tokenisers = self.debug or self.span_kd_enabled
        if need_tokenisers and tokenizer_student is not None:
            self.student_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_student, trust_remote_code=True
            )
        if need_tokenisers and tokenizer_teacher is not None:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_teacher, trust_remote_code=True
            )

        self.store_teacher_logits = True
        self.batch_limit = batch_limit  # 设定每100个样本保存一次
        self.store_path = store_path
        self.teacher_logits_temp_storage = []

        if self.debug:
            print("Distillation loss parameters:")
            print(f"Crossentropy weight: {crossentropy_weight}")
            print(f"Distillation weight: {distillation_weight}")
            print(f"Student temperature: {student_temperature}")
            print(f"Teacher temperature: {teacher_temperature}")
            print(f"Skip student eos: {skip_student_eos}")
            print(f"Skip teacher eos: {skip_teacher_eos}")
            print(f"Ignore index: {ignore_index}")
            print(f"Debug: {debug}")
            print(f"Debug rank: {debug_rank}")
        if self.span_kd_enabled:
            print(
                f"[SpanOT-KD] enabled — aggregation={self.span_aggregation}, "
                f"top_r={self.span_top_r}, low_delta={self.span_low_delta}"
            )

    def forward(self, epoch, student_predictions, teacher_predictions, student_targets, teacher_targets, rank=0, step=None):
        student = student_predictions.logits
        teacher = teacher_predictions.logits
        if self.store_teacher_logits:
            processed_teacher_logits = []

        # Get answer first token and answer size
        student_answer_index, student_answer_size = self.__get_start_and_size_answers(
            student_targets)
        teacher_answer_index, teacher_answer_size = self.__get_start_and_size_answers(
            teacher_targets)

        # Avoid eos token, if needed
        if self.skip_student_eos: student_answer_size = [size-1 for size in student_answer_size]
        if self.skip_teacher_eos: teacher_answer_size = [size-1 for size in teacher_answer_size]
        
        student = normalize(student)      
        teacher = normalize(teacher)

        # Align answer first token, pad to right and compute softmax
        for i in range(student.size(0)):
            shift = student_answer_index[i]
            size = student_answer_size[i]
            end_shift = shift+size
            student[i] = torch.cat((
                torch.nn.functional.softmax(student[i, shift:end_shift, :]/self.student_temperature, dim=-1),
                torch.zeros_like(student[i, :(student.size(1)-size), :])), dim=0
            )
        for i in range(teacher.size(0)):
            shift = teacher_answer_index[i]
            size = teacher_answer_size[i]
            end_shift = shift+size
            teacher[i] = torch.cat((
               torch.nn.functional.softmax(teacher[i, shift:end_shift, :]/self.teacher_temperature, dim=-1),
               torch.zeros_like(teacher[i, :(teacher.size(1)-size), :])), dim=0
            )
        
        # Cut to max answer length
        mex_length = max(max(student_answer_size), max(teacher_answer_size))

        student = student[:, :mex_length, :]
        teacher = teacher[:, :mex_length, :]
        
        sinkorn_loss = Sinkhorn_seq()

        if self.debug and rank == self.debug_rank:
            print("\n\n----------------------------------")
            print("------- Label / Prediction -------")
            print("----------------------------------")
            student_labels = [row[row != -100] for row in student_targets]
            teacher_labels = [row[row != -100] for row in teacher_targets]
            print("------- Label shape -------")
            print(f"Student label shape: {student_answer_size[0]}")
            print(f"Teacher label shape: {teacher_answer_size[0]}")
            print("------- Student Label -> Prediction -------")
            print(self.student_tokenizer.batch_decode(student_labels[0]))
            print(self.student_tokenizer.batch_decode(torch.argmax(
                student[0][:student_answer_size[0]], dim=-1)))
            print("------- Teacher Label -> Prediction -------")
            print(self.teacher_tokenizer.batch_decode(teacher_labels[0]))
            print(self.teacher_tokenizer.batch_decode(torch.argmax(
                teacher[0][:teacher_answer_size[0]], dim=-1)))
            print("------- Prediction Teacher -> Student  -------")
            
            print(self.teacher_tokenizer.batch_decode(torch.argmax(
                teacher[0][:teacher_answer_size[0]], dim=-1)))
            print(self.student_tokenizer.batch_decode(torch.argmax(
                student[0][:student_answer_size[0]], dim=-1)))
            print("------- Shape -------")
            print(f"Student shape: {student.size()}")
            print(f"Teacher shape: {teacher.size()}\n")

        # # Sort in descending order to align probabilities
        # student = student.sort(dim=-1, descending=True).values
        # teacher = teacher.sort(dim=-1, descending=True).values
        teacher = improved_sort(teacher)
        teacher = teacher[:,:,:50]
        if self.f == 1:
            student = improved_sort(student)
            student = student[:,:,:50]
        elif self.f == 2:
            student = greedy_algorithm_adjust_s(teacher,student)

        # Pad to get same vocabulary size
        diff_size = student.size(2) - teacher.size(2)
        if diff_size > 0:
            teacher = F.pad(teacher, (0, diff_size), value=0)
        elif diff_size < 0:
            student = F.pad(student, (0, abs(diff_size)), value=0)
            
        if self.debug and rank == self.debug_rank:
            print("--------------------------------------------")
            print("---- Post-treatment tensor architecture ----")
            print("--------------------------------------------")
            print("------- Shape -------")
            print(f"Student shape: {student.size()}")
            print(f"Teacher shape: {teacher.size()}")
            print(" ------- First token -------")
            print(f"Student first logits: {student[0][0][:5].tolist()}")
            print(f"Teacher first logits: {teacher[0][0][:5].tolist()}")
            print(f"Student last logits: {student[0][0][-5:].tolist()}")
            print(f"Teacher last logits: {teacher[0][0][-5:].tolist()}")
            print(" ------- Last token -------")
            print(f"Student first logits: {student[0][-1][:5].tolist()}")
            print(f"Teacher first logits: {teacher[0][-1][:5].tolist()}")
            print(f"Student last logits: {student[0][-1][-5:].tolist()}")
            print(f"Teacher last logits: {teacher[0][-1][-5:].tolist()}\n")

        # Cross entropy loss
        crossentropy_loss = self.crossentropy_weight * student_predictions.loss

        # --- SpanOT-KD: compute per-position weight tensor ---
        position_weights = None
        if self.span_kd_enabled and self.student_tokenizer is not None and self.teacher_tokenizer is not None:
            from train.span_ot import compute_batch_position_weights
            position_weights = compute_batch_position_weights(
                student_probs=student.detach(),
                teacher_probs=teacher.detach(),
                student_sizes=student_answer_size,
                teacher_sizes=teacher_answer_size,
                student_labels=student_targets,
                teacher_labels=teacher_targets,
                student_tokenizer=self.student_tokenizer,
                teacher_tokenizer=self.teacher_tokenizer,
                top_r=self.span_top_r,
                low_delta=self.span_low_delta,
                aggregation=self.span_aggregation,
                ignore_index=self.ignore_index,
            )

        # --- SpanOT-KD side analysis (read-only, env-gated) ---
        # When SPANOT_TRACE is unset this is a single env read and a no-op, so
        # the loss math above is unchanged. When set on rank 0 at the configured
        # (step, b) it dumps every intermediate of the weight construction for
        # one deterministic sample (see train.span_trace).
        if self.span_kd_enabled and self.student_tokenizer is not None and self.teacher_tokenizer is not None:
            from train.span_trace import should_trace, trace_sample
            _trace = should_trace(step, rank)
            if _trace is not None:
                trace_b, _ = _trace
                if 0 <= trace_b < student.size(0):
                    try:
                        trace_sample(
                            student_probs=student[trace_b].detach(),
                            teacher_probs=teacher[trace_b].detach(),
                            student_raw_logits=student_predictions.logits[trace_b].detach(),
                            teacher_raw_logits=teacher_predictions.logits[trace_b].detach(),
                            student_size=int(student_answer_size[trace_b]),
                            teacher_size=int(teacher_answer_size[trace_b]),
                            student_answer_index=int(student_answer_index[trace_b]),
                            teacher_answer_index=int(teacher_answer_index[trace_b]),
                            student_label_row=student_targets[trace_b],
                            teacher_label_row=teacher_targets[trace_b],
                            student_tokenizer=self.student_tokenizer,
                            teacher_tokenizer=self.teacher_tokenizer,
                            student_temperature=self.student_temperature,
                            teacher_temperature=self.teacher_temperature,
                            top_r=self.span_top_r,
                            low_delta=self.span_low_delta,
                            aggregation=self.span_aggregation,
                            ignore_index=self.ignore_index,
                            meta={"seed": os.environ.get("SPANOT_TRACE_SEED"),
                                  "step": int(step), "b": int(trace_b), "epoch": int(epoch)},
                        )
                    except Exception as _e:
                        print(f"[SpanOT-trace] failed (non-fatal): {_e}", flush=True)

        # --- Distillation loss components ---
        # (1) L1 / OT-style component: per-sample |p_s - p_t| summed over vocab, averaged over tokens.
        # SpanOT-KD weights each token-position contribution by omega(t) before the length-mean  
        #  With `position_weights=None` the math reduces
        # exactly to the original `.sum(-1).mean(-1)` formulation.
        l1_per_sample = torch.zeros(student.size(0), device=student.device)
        for i in range(student.size(0)):
            size = min(student_answer_size[i], teacher_answer_size[i])
            if size <= 0:
                continue
            per_pos_l1 = abs(student[i][:size] - teacher[i][:size]).sum(-1)  # (size,)
            if position_weights is not None:
                w_i = position_weights[i, :size].to(per_pos_l1.dtype)
                per_pos_l1 = per_pos_l1 * w_i
            l1_per_sample[i] = per_pos_l1.mean()
        l1_component = l1_per_sample.mean()

        # (2) KL/SL-style component (scalar, weighted by 0.1 as in original formulation).
        # SpanOT-KD: position-level weights are passed into KL_wo so each
        # token's CE contribution is scaled before the batch+position mean
        # (Eq. 16 with `omega(t)` applied per (B, T) position).
        kl_component = KL_wo(teacher, student, position_weights=position_weights) * 0.1

        # (3) Sinkhorn (Sequence Distance) component. SpanOT-KD applies row-wise
        # weighting AFTER the Sinkhorn normalisation (Eq. 17): the transport
        # plan P stays identical to vanilla MultiLevelOT and only the loss
        # aggregation `sum_{i,j} W_{ij} * P_{ij} * C^seq_{ij}` is reweighted.
        # `position_weights` (B, T) is broadcast to (B, T, T) row-wise inside
        # `Sinkhorn_seq.sinkhorn_loss`.
        sinkhorn_epsilon = 0.1  # entropic-regularisation strength (paper: lambda)
        sinkhorn_gamma = 0.1    # external weight applied in THIS file (paper: gamma)
        sinkhorn_raw_tensor, sinkhorn_diag = sinkorn_loss.forward_with_diagnostics(
            teacher, student, epsilon=sinkhorn_epsilon, row_weights=position_weights,
        )
        sinkhorn_component = sinkhorn_raw_tensor * sinkhorn_gamma
       
        if isinstance(sinkhorn_raw_tensor, torch.Tensor):
            sinkhorn_raw_value = float(sinkhorn_raw_tensor.detach().item())
        else:
            sinkhorn_raw_value = float(sinkhorn_raw_tensor)

        l1_component = self.distillation_weight * l1_component
        kl_component = self.distillation_weight * kl_component
        sinkhorn_component = self.distillation_weight * sinkhorn_component

        distillation_loss = l1_component + kl_component + sinkhorn_component

        diagnostics = {
            "sinkhorn_raw_value": sinkhorn_raw_value,
            "sinkhorn_per_sample_raw_mean": sinkhorn_diag["sinkhorn_per_sample_raw_mean"],
            "sinkhorn_epsilon": sinkhorn_diag["sinkhorn_epsilon"],
            "sinkhorn_gamma": sinkhorn_gamma,
            "sinkhorn_distillation_weight": float(self.distillation_weight),
            "cost_mean": sinkhorn_diag["cost_mean"],
            "cost_max": sinkhorn_diag["cost_max"],
            "cost_min": sinkhorn_diag["cost_min"],
            "span_kd_enabled": bool(self.span_kd_enabled),
            "span_top_r": float(self.span_top_r),
            "span_low_delta": float(self.span_low_delta),
            "span_aggregation": str(self.span_aggregation),
        }
        if position_weights is not None:
            # Fraction of (B, T_max) positions that are downweighted vs at unit
            # weight — useful sanity check that `top_r` is producing a non-trivial
            # partition. Positions strictly < 1.0 are low-priority spans (delta);
            # positions == 1.0 are either high-priority or non-matched.
            with torch.no_grad():
                w = position_weights
                low_frac = float((w < 1.0).float().mean().item())
                mean_w = float(w.mean().item())
            diagnostics["span_low_weight_frac"] = low_frac
            diagnostics["span_position_weight_mean"] = mean_w
        else:
            diagnostics["span_low_weight_frac"] = 0.0
            diagnostics["span_position_weight_mean"] = 1.0

        if self.debug and rank == self.debug_rank:
            print("--------------------------------------")
            print("---------------- Loss ----------------")
            print("--------------------------------------")
            print(f"Crossentropy loss: {crossentropy_loss}")
            print(f"Distillation loss: {distillation_loss}")
            print(f"  L1/OT component:   {l1_component}")
            print(f"  KL component:      {kl_component}")
            print(f"  Sinkhorn comp.:    {sinkhorn_component}")
            print(f"  [diag] sinkhorn raw (pre-gamma, pre-w): {sinkhorn_raw_value:.6e}")
            print(f"  [diag] sinkhorn per-sample raw mean:    {diagnostics['sinkhorn_per_sample_raw_mean']:.6e}")
            print(f"  [diag] cost C  mean={diagnostics['cost_mean']:.6e} "
                  f"max={diagnostics['cost_max']:.6e} min={diagnostics['cost_min']:.6e} "
                  f"(epsilon={sinkhorn_epsilon})")
            print(f"Total loss: {crossentropy_loss + distillation_loss}")

        return (
            crossentropy_loss + distillation_loss,
            crossentropy_loss,
            distillation_loss,
            l1_component,
            kl_component,
            sinkhorn_component,
            diagnostics,
        )

    def __get_start_and_size_answers(self, answer_tensors):
        answers_index = []
        answers_size = []

        for answer in answer_tensors:
            is_value = answer.eq(self.ignore_index)
            answers_size.append(len(answer) - int(is_value.sum()))
            indices = is_value.nonzero(as_tuple=True)[0]
            if len(indices) == 0 or indices[0] != 0:
                answers_index.append(0)
            else:
                diff_indices = indices[1:] - indices[:-1]
                break_index = (diff_indices != 1).nonzero()
                length = (break_index[0].item() +
                          1) if len(break_index) > 0 else len(indices)
                answers_index.append(length-1)
        return answers_index, answers_size
    
    def save_teacher_logits_partial(self):

        with open(self.store_path, 'ab') as f:  
            for logits in self.teacher_logits_temp_storage:
                np.save(f, logits)

    def on_epoch_end(self):

        if self.teacher_logits_temp_storage:
            self.save_teacher_logits_partial()
            self.teacher_logits_temp_storage = []  
