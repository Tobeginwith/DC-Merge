import torch
from collections import defaultdict, OrderedDict
from copy import deepcopy
from tqdm.auto import tqdm


def tv_merging(vectors, weights=None, merging_type='unmerged', **kwargs):

    rows_to_keep = [torch.ones_like(vectors[i]) for i in range(len(vectors))]
    vectors_ = torch.vstack(vectors).clone()
    if merging_type == 'unmerged':
        return vectors_, rows_to_keep
    
    if weights is not None:
        for vector in vectors_:
            vector *= weights[0]
    if merging_type == 'mean':
        result = torch.mean(vectors_, dim=0)
    else:
        result = torch.sum(vectors_, dim=0)
    return result, rows_to_keep, rows_to_keep


## TIES MERGING UTILS
def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100
    
    if K >= 1 and return_mask:
        return M, torch.ones_like(M).mean(dim=-1), torch.ones_like(M)
    elif K >= 1:
        return M, torch.ones_like(M).mean(dim=-1)

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements
    # Find the k-th smallest element by magnitude for each row
    if M.flatten().shape[-1] == 1:
        kth_values = M.abs()
    else:
        kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    if original_shape == M.squeeze().shape:
        final_mask = mask.squeeze()
        M = M.squeeze()
    else:
        final_mask = mask
        
    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=-1), final_mask
    return M * final_mask, final_mask.float().mean(dim=-1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def chunked_disjoint_mean(vectors, chunk_size=10000):
    num_chunks = vectors.size(0) // chunk_size + (1 if vectors.size(0) % chunk_size != 0 else 0)
    total_sum = torch.zeros_like(vectors[0])
    non_zero_counts = torch.zeros_like(vectors[0])

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, vectors.size(0))
        chunk = vectors[start_idx:end_idx]

        # Calculate sum and non-zero counts for the chunk
        total_sum += torch.sum(chunk, dim=0)
        non_zero_counts += (chunk != 0).sum(dim=0)

    # Compute the disjoint mean
    disjoint_aggs = total_sum / torch.clamp(non_zero_counts.float(), min=1)
    disjoint_aggs[non_zero_counts == 0] = 0

    return disjoint_aggs


def chunked_sum(tensor, chunk_size=10000):
    num_chunks = tensor.size(0) // chunk_size + (1 if tensor.size(0) % chunk_size != 0 else 0)
    total_sum = torch.zeros_like(tensor[0])

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, tensor.size(0))
        chunk = tensor[start_idx:end_idx]

        # Add the sum of the current chunk to the total sum
        total_sum += torch.sum(chunk, dim=0)

    return total_sum


def disjoint_merge(Tensor, merge_func, reference_sign_to_mult, weights=None):
    # If sign is provided then we select the corresponding entries and aggregate.
    if reference_sign_to_mult is not None:
        rows_to_keep = torch.where(
            reference_sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
        )
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = Tensor != 0
        
    selected_entries = Tensor * rows_to_keep
    if weights is not None:
        for selected_entrie in selected_entries:
            selected_entrie *= weights[0]
    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = chunked_sum(selected_entries)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= reference_sign_to_mult
    elif merge_func == 'unmerged':
        disjoint_aggs = selected_entries
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs, rows_to_keep

def resolve_sign(Tensor, mode=None):
    sign_to_mult = torch.sign(Tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult

def ties_merging(vectors, topK=10, merging_type='mean', weights=None, **kwargs):
    # Add functionality that allows some layers to not be pruned or lets them be skipped
    print(f'TopK is: {topK}')
    print(f'weights is: {weights}')
    stacked_vectors = torch.vstack(vectors).clone()
    pruned_vectors, _, mask = topk_values_mask(
        stacked_vectors, K=topK, return_mask=True
    )
    vector_signs = resolve_sign(pruned_vectors)
    assert vector_signs is not None
    merged_tv, rows_to_keep = disjoint_merge(pruned_vectors, merging_type, vector_signs, weights)
    return merged_tv, rows_to_keep, mask

def generate_linear_distribution(num_classes, ratio):
    s = torch.linspace(ratio, 1.0, num_classes)
    s = s / s.sum()
    return s


def dc_merge(deltas_dict, smoothing_strategy='avg', rho=5.0):
    dc_dict = {}


    for k, vecs in tqdm(deltas_dict.items(), desc='DC-Merge Processing...'):
        N = len(vecs)
        low_rank_per_task = 16
        smoothed_vecs = []
        
        for i in range(N):
            u, s, v = torch.linalg.svd(vecs[i], full_matrices=False)
            if i == 0:
                sum_u = torch.zeros_like(u)
                sum_v = torch.zeros_like(v)

            sum_u[:, i * low_rank_per_task : (i + 1) * low_rank_per_task] = u[:, :low_rank_per_task]
            sum_v[i * low_rank_per_task : (i + 1) * low_rank_per_task, :] = v[:low_rank_per_task, :]
            
            orig_energy = s[:low_rank_per_task].clone()
            if smoothing_strategy == 'linear':
                smoothed_ratio = min(rho, orig_energy[0] / orig_energy[-1])
                smoothed_energy_dist = generate_linear_distribution(low_rank_per_task, smoothed_ratio)
            elif smoothing_strategy == 'avg':
                smoothed_energy_dist = torch.ones_like(orig_energy) / len(orig_energy)
            else:
                raise ValueError("Invalid smoothing strategy")
            
            smoothed_energy = orig_energy.sum() * smoothed_energy_dist
            smoothed_vecs.append(u[:, :low_rank_per_task] @ torch.diag(smoothed_energy) @ v[:low_rank_per_task, :])
           
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        cover_space_u = (u_u @ v_u)[:, :N * low_rank_per_task]
        cover_space_vT = (u_v @ v_v)[:N * low_rank_per_task, :]

        Ms = [torch.linalg.multi_dot((cover_space_u.T, smoothed_vecs[i], cover_space_vT.T, )) for i in range(N)]
        filtered_Ms = keep_topk_percent(Ms, 1e-3)
        agg_M = ties_small(filtered_Ms)
        mask_M = torch.zeros_like(agg_M)
        d_per_task = mask_M.shape[0] // N
        for i in range(N):
            mask_M[i * d_per_task : (i+1) * d_per_task, i * d_per_task : (i+1) * d_per_task] = 1
        
        dc_dict[k] = torch.linalg.multi_dot((cover_space_u, agg_M * mask_M , cover_space_vT, ))

    return OrderedDict(sorted(dc_dict.items()))


def keep_topk_percent(tensor_list, percent=0.1):
    new_list = []
    for i, t in enumerate(tensor_list):
        flat_abs = t.abs().view(-1)
        k = max(1, int(percent * flat_abs.numel()))
        threshold = torch.topk(flat_abs, k).values.min()
        mask = t.abs() >= threshold
        new_t = t * mask
        new_list.append(new_t)
    
    return new_list


def ties_small(mat_list):
    stacked = torch.stack(mat_list, dim=0)
    summed = stacked.sum(dim=0)
    summed_sign = torch.sign(summed)
    elem_sign = torch.sign(stacked)
    mask = (elem_sign == summed_sign.unsqueeze(0))
    count = mask.sum(dim=0)
    selected = stacked * mask
    res = selected.sum(dim=0) / count.clamp(min=1)
    res = torch.where(summed == 0, torch.zeros_like(res), res)

    return res


def get_redundant_task_vector(vectors, iter_num=300):
    vectors = vectors.cuda()
    print(vectors.shape)
    merging_vector = torch.nn.Parameter((torch.sum(vectors, dim=0)))
    optimizer = torch.optim.Adam([merging_vector], lr=1e-5, weight_decay=0)
    l2_norms = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1))

    for i in tqdm(range(iter_num)):
        disturbing_vectors = merging_vector.unsqueeze(0) - vectors
        inner_product = torch.matmul(disturbing_vectors , vectors.transpose(1,2))

        loss = torch.sum(torch.square(inner_product) / l2_norms.unsqueeze(-1).unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return merging_vector.data.detach().cpu()


def wudi(deltas_dict, iter_num=300):
    merged_task_vector = {}

    for k, mat_list in tqdm(deltas_dict.items(), desc='WUDI Processing...'):
        values = deepcopy(torch.stack(mat_list, dim=0))
        merigng_vector = get_redundant_task_vector(values, iter_num)
        merged_task_vector[k] = merigng_vector

    return OrderedDict(sorted(merged_task_vector.items()))


def iso_cts(deltas_dict, k_frac=0.8):
    iso_dict = {}

    for key, mat_list in tqdm(deltas_dict.items(), desc='Iso-CTS Processing...'):
        N = len(mat_list)
        delta_TA = torch.stack(mat_list, dim=0).sum(dim=0)
        u, s, v = torch.linalg.svd(delta_TA, full_matrices=False)
        low_rank_per_task = 16
        common_space_index_s = int(N * low_rank_per_task * k_frac)
        _task_specific_total_space_index_s = round((N * low_rank_per_task - common_space_index_s) / N) * N
        common_space_index_s = N * low_rank_per_task - _task_specific_total_space_index_s

        u_mixed = torch.zeros_like(u)
        common_space_u = u[:, :common_space_index_s]

        s_mixed = torch.zeros_like(s)
        common_space_s = s[:common_space_index_s]

        v_mixed = torch.zeros_like(v)
        common_space_v = v[:common_space_index_s, :]

        n_dims_per_task = max(1, int((N * low_rank_per_task - common_space_index_s) / N))
        for i in range(N):
            delta_hat = mat_list[i] - torch.linalg.multi_dot((common_space_u, common_space_u.T, mat_list[i],))
            u_hat, s_hat, v_hat = torch.linalg.svd(delta_hat, full_matrices=False)

            u_mixed[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_hat[:, :n_dims_per_task]
            s_mixed[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_hat[:n_dims_per_task]
            v_mixed[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_hat[:n_dims_per_task, :]

        u_mixed[:, N * n_dims_per_task : N * n_dims_per_task + common_space_index_s] = common_space_u
        s_mixed[N * n_dims_per_task : N * n_dims_per_task + common_space_index_s] = common_space_s
        v_mixed[N * n_dims_per_task : N * n_dims_per_task + common_space_index_s, :] = common_space_v

        u_u, s_u, v_u = torch.linalg.svd(u_mixed, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(v_mixed, full_matrices=False)
        
        U = (u_u @ v_u)[:, :N * low_rank_per_task]
        V = (u_v @ v_v)[:N * low_rank_per_task, :]
        s_mixed = s_mixed[:N * low_rank_per_task]
        s_bar = s_mixed.mean() * torch.ones_like(s_mixed)

        iso_dict[key] = torch.linalg.multi_dot(
            (
                U,
                torch.diag(s_bar),
                V,
            )
        )

    return OrderedDict(sorted(iso_dict.items()))


def TSVM(deltas_dict):
    tsvm_dict = {}
    for k, mat_list in tqdm(deltas_dict.items(), desc='TSVM Processing...'):
        N = len(mat_list)
        for i in range(N):
            u, s, v = torch.linalg.svd(mat_list[i], full_matrices=False)
            if i == 0:
                sum_u = torch.zeros_like(u)
                sum_s = torch.zeros_like(s)
                sum_v = torch.zeros_like(v)
            reduced_index_s = 16

            sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[:, :reduced_index_s]
            sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:reduced_index_s]
            sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[:reduced_index_s, :]
        
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        U = (u_u @ v_u)[:, :N * reduced_index_s]
        V = (u_v @ v_v)[:N * reduced_index_s, :]
        
        tsvm_dict[k] = torch.linalg.multi_dot(
            (
                U,
                torch.diag(sum_s[:N * reduced_index_s]),
                V,
            )
        )
    
    return OrderedDict(sorted(tsvm_dict.items()))


def TA(deltas_dict):
    ta_dict = {}
    for k, mat_list in tqdm(deltas_dict.items(), desc='TA Processing...'):
        res = torch.sum(torch.stack(mat_list, dim=0), dim=0)
        ta_dict[k] = res

    return OrderedDict(sorted(ta_dict.items()))
