"""Training routines: seed fine-tuning, teacher adaptation, batched distillation + promotion (Step 9)."""
from __future__ import annotations
import math, random
import numpy as np
import torch

from .losses import distill_loss
from .metrics import classification_metrics


def _batches(items, bs, shuffle=True, seed=0):
    items = list(items)
    if shuffle:
        random.Random(seed).shuffle(items)
    for s in range(0, len(items), bs):
        yield items[s:s + bs]


def train_examples(vlm, examples, load_image, classes, epochs=1, lr=1e-4, batch_size=2, grad_accum=4,
                   T_d=2.0, lam=0.7, mu=0.5, seed=0, log=print):
    """Generic LoRA training loop over example dicts:
       {path, y_star, z_cloud (list|None), plan (str|None), weight, teacher_wrong}."""
    params = vlm.trainable_parameters()
    if not params:
        raise RuntimeError("Model has no trainable (LoRA) parameters")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = math.ceil(len(examples) / batch_size) * epochs // grad_accum + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 2), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    step, running, pending = 0, [], False
    with vlm.training():
        for ep in range(epochs):
            for i, batch in enumerate(_batches(examples, batch_size, seed=seed + ep)):
                ims = [load_image(e["path"]) for e in batch]
                names = [classes[e["y_star"]] for e in batch]
                plans = [e.get("plan") or None for e in batch]
                z_local, plan_nll = vlm.forward_for_training(ims, names, plans)
                dev = z_local.device
                y = torch.tensor([e["y_star"] for e in batch], device=dev)
                w = torch.tensor([float(e.get("weight", 1.0)) for e in batch], device=dev)
                tw = torch.tensor([bool(e.get("teacher_wrong", False)) for e in batch], device=dev)
                zc = None
                if all(e.get("z_cloud") is not None for e in batch):
                    zc = torch.tensor([e["z_cloud"] for e in batch], device=dev, dtype=torch.float32)
                loss = distill_loss(z_local, zc, y, plan_nll, w, tw, T_d=T_d, lam=lam, mu=mu) / grad_accum
                scaler.scale(loss).backward()
                running.append(loss.item() * grad_accum)
                pending = True
                if (i + 1) % grad_accum == 0:
                    pending = False
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                    if step < sched.total_steps - 1:
                        sched.step()
                    step += 1
                    if step % 20 == 0:
                        log(f"  epoch {ep} step {step} loss {np.mean(running[-20 * grad_accum:]):.4f}")
            if pending:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    return float(np.mean(running)) if running else float("nan")


def evaluate(vlm, paths, y_true, load_image, T=1.0, no_tumour_idx=3, batch_size=8):
    from .gate import softmax
    z = vlm.class_logits([load_image(p) for p in paths], batch_size=batch_size)
    p = softmax(z, T)
    m = classification_metrics(y_true, p.argmax(1), p, no_tumour_idx)
    return m, z


def mcnemar_worse_p(cur_correct, new_correct) -> float:
    """One-sided exact McNemar test: probability of seeing at least this many 'old right, new wrong'
    images among the discordant ones if the two models were equally good. Small p = new model is worse."""
    from math import comb
    cur_correct, new_correct = np.asarray(cur_correct, bool), np.asarray(new_correct, bool)
    b = int((cur_correct & ~new_correct).sum())      # old right, new wrong
    c = int((~cur_correct & new_correct).sum())      # old wrong, new right
    n = b + c
    if n == 0:
        return 1.0
    return float(sum(comb(n, k) for k in range(b, n + 1)) / 2 ** n)


def promotion_decision(cur_m, new_m, cur_correct, new_correct, cfg_d):
    """v1.9 promotion rule: reject only if the candidate is significantly worse on validation accuracy
    (McNemar, alpha) or loses more than sens_tol of tumour sensitivity (safety guard)."""
    p = mcnemar_worse_p(cur_correct, new_correct)
    alpha = float(cfg_d.get("promotion_alpha", 0.05))
    sens_tol = float(cfg_d.get("sens_tol", 0.01))
    ok = p >= alpha and new_m["tumour_sensitivity"] >= cur_m["tumour_sensitivity"] - sens_tol
    return ok, p


def distill_round_cumulative(student, memory, replay_pool, val_paths, val_y, load_image, classes, cfg,
                             current_val=None, current_correct=None, seed=0, log=print):
    """v1.9 online distillation round: warm-start from the deployed adapter and train on ALL verified labels
    collected so far (memory) plus replayed seed examples; promote unless significantly worse."""
    d = cfg.distill
    nt = classes.index("no_tumour")
    before = student.adapter_state()
    if current_val is None or current_correct is None:
        current_val, z = evaluate(student, val_paths, val_y, load_image, no_tumour_idx=nt)
        current_correct = z.argmax(1) == np.asarray(val_y)
    n_rep = int(len(memory) * d.replay_ratio)
    replay = random.Random(seed).sample(replay_pool, min(n_rep, len(replay_pool)))
    train_set = list(memory) + replay
    loss = train_examples(student, train_set, load_image, classes, epochs=d.get("cum_epochs", 1),
                          lr=d.get("cum_lr", d.lr), T_d=d.T_d, lam=d.lam, mu=d.mu, seed=seed, log=log)
    new_val, z = evaluate(student, val_paths, val_y, load_image, no_tumour_idx=nt)
    new_correct = z.argmax(1) == np.asarray(val_y)
    ok, p = promotion_decision(current_val, new_val, current_correct, new_correct, d)
    info = dict(loss=loss, train_size=len(train_set), p_worse=p, val_candidate=new_val)
    if not ok:
        student.load_adapter_state(before)
        log(f"  round rejected (val acc {new_val['accuracy']:.3f} vs {current_val['accuracy']:.3f}, "
            f"p_worse={p:.3f}, sens {new_val['tumour_sensitivity']:.3f}); rolled back")
        return dict(info, promoted=False, val=current_val, correct=current_correct)
    log(f"  round promoted on {len(train_set)} examples (val acc {current_val['accuracy']:.3f} -> "
        f"{new_val['accuracy']:.3f}, F1 {current_val['macro_f1']:.3f} -> {new_val['macro_f1']:.3f}, p_worse={p:.3f})")
    return dict(info, promoted=True, val=new_val, correct=new_correct)


def distill_round(student, buffer, replay_pool, val_paths, val_y, load_image, classes, cfg, current_val=None,
                  seed=0, log=print):
    """One distillation event: train on buffer + replay, validate, promote or roll back (Section 3.4.9)."""
    d = cfg.distill
    before = student.adapter_state()
    if current_val is None:
        current_val, _ = evaluate(student, val_paths, val_y, load_image, no_tumour_idx=classes.index("no_tumour"))
    n_rep = int(len(buffer) * d.replay_ratio)
    replay = random.Random(seed).sample(replay_pool, min(n_rep, len(replay_pool)))
    loss = train_examples(student, list(buffer) + replay, load_image, classes, epochs=d.epochs, lr=d.lr,
                          T_d=d.T_d, lam=d.lam, mu=d.mu, seed=seed, log=log)
    new_val, _ = evaluate(student, val_paths, val_y, load_image, no_tumour_idx=classes.index("no_tumour"))
    eps = d.eps_regression
    ok = (new_val["macro_f1"] >= current_val["macro_f1"] - eps and
          new_val["tumour_sensitivity"] >= current_val["tumour_sensitivity"] - eps)
    if not ok:
        student.load_adapter_state(before)
        log(f"  round rejected (val F1 {new_val['macro_f1']:.3f} vs {current_val['macro_f1']:.3f}); rolled back")
        return {"promoted": False, "loss": loss, "val": current_val, "val_candidate": new_val}
    log(f"  round promoted (val F1 {current_val['macro_f1']:.3f} -> {new_val['macro_f1']:.3f})")
    return {"promoted": True, "loss": loss, "val": new_val, "val_candidate": new_val}
