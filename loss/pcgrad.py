import torch
from torch import Tensor


def pcgrad(task_grads: list[dict[str, Tensor]]) -> tuple[dict[str, Tensor], float]:
    """
    PCGrad surgery on encoder gradients.

    For each task pair with conflicting gradients (negative dot product),
    projects out the conflicting component before summing.

    task_grads: list of {param_name: grad_tensor}, one dict per task
    Returns: (summed modified grads dict, conflict_rate float)
    """
    if len(task_grads) < 2:
        return task_grads[0], 0.0

    keys = list(task_grads[0].keys())
    # Build per-task flat gradient vectors (only keys present in all tasks)
    shared_keys = [k for k in keys if all(k in g for g in task_grads)]
    if not shared_keys:
        return task_grads[0], 0.0

    flat = [torch.cat([g[k].flatten() for k in shared_keys]) for g in task_grads]

    modified_flat = []
    conflicts = 0
    total_pairs = 0
    for i, g_i in enumerate(flat):
        g_i = g_i.clone()
        for j, g_j in enumerate(flat):
            if i == j:
                continue
            dot = torch.dot(g_i, g_j)
            total_pairs += 1
            if dot < 0:
                g_i -= (dot / (g_j.norm() ** 2 + 1e-8)) * g_j
                conflicts += 1
        modified_flat.append(g_i)

    conflict_rate = conflicts / max(total_pairs, 1)

    summed_flat = sum(modified_flat)
    result = {}
    offset = 0
    for k in shared_keys:
        sz = task_grads[0][k].numel()
        result[k] = summed_flat[offset:offset + sz].view_as(task_grads[0][k])
        offset += sz

    return result, conflict_rate
