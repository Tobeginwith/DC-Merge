import torch
from tqdm.auto import tqdm

def dc_merge(task_vectors, config):
    """
    DC-Merge
    """
    sv_reduction = 1 / len(config.DATASETS)
    device = config.device
    print("Computing DC-Merge...")

    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            trunc_vecs = []
            for i, (task_vector, dataset) in enumerate(zip(task_vectors, config.DATASETS)):
                vec = task_vector.vector[key].to(device)

                if (len(task_vector.vector[key].shape) == 2) and ("text_projection" not in key):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                    low_rank_per_task = int(s.shape[0] * sv_reduction)

                    sum_u[:, i * low_rank_per_task : (i + 1) * low_rank_per_task] = u[:, :low_rank_per_task]
                    sum_v[i * low_rank_per_task : (i + 1) * low_rank_per_task, :] = v[:low_rank_per_task, :]
                    
                    trunc_recon = u[:, :low_rank_per_task] @ torch.diag(s[:low_rank_per_task]) @ v[:low_rank_per_task, :]
                    trunc_vecs.append(trunc_recon)

                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                
                cover_space_u = u_u @ v_u
                cover_space_vT = u_v @ v_v
                
                Ms = [torch.linalg.multi_dot((cover_space_u.T, trunc_vecs[i].to(device), cover_space_vT.T, )) for i in range(len(config.DATASETS))]
                filtered_Ms = keep_topk_percent(Ms, 1e-3)
                agg_M = ties_small(filtered_Ms)
                mask_M = torch.zeros_like(agg_M)
                d_per_task = mask_M.shape[0] // len(config.DATASETS)
                for i in range(len(config.DATASETS)):
                    mask_M[i * d_per_task : (i+1) * d_per_task, i * d_per_task : (i+1) * d_per_task] = 1
                
                new_vector[key] = torch.linalg.multi_dot((cover_space_u, agg_M * mask_M, cover_space_vT, ))
                
    return new_vector


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


def iso_cts(task_vectors, config, common_space_fraction=0.8):
    device = config.device
    new_vector = {}

    print("Computing Iso-CTS...")
    for key in task_vectors[0].vector:
        shape_ = task_vectors[0].vector[key].shape

        is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
        if not is_2d_matrix:
            print(f"Combining by avg {key}...")
            for i, (task_vector, dataset) in enumerate(zip(task_vectors, config.DATASETS)):
                vec = task_vector.vector[key].to(device)
                if i == 0:
                    new_vector[key] = vec.clone()
                else:
                    new_vector[key] += (vec - new_vector[key]) / (i + 1)
            continue
        
        print(f"Computing common space using sum for {key}...")
        combined_w = sum([task_vector.vector[key].to(device) for task_vector in task_vectors])

        ### Calculate the common space size (making sure that task specific space is equally divisible) ###
        common_space_index_s = int(min(shape_) * common_space_fraction)
        _task_specific_total_space_index_s = round((min(shape_) - common_space_index_s) / len(config.DATASETS)) * len(config.DATASETS)
        common_space_index_s = min(shape_) - _task_specific_total_space_index_s

        u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
        common_space_u = u[:, :common_space_index_s]
        common_space_s = s[:common_space_index_s]
        common_space_v = v[:common_space_index_s, :]
        ###################################################################
        
        ### Calculate task specific space ###
        n_dims_per_task = int((min(shape_) - common_space_index_s) / len(config.DATASETS))
        for i, task_vector in enumerate(task_vectors):
            w = task_vector.vector[key].to(device)

            # calculate the projection onto task specific space to remove the common space
            w_ts = w - common_space_u @ common_space_u.T @ w
            u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)            
            
            if i == 0:
                combined_space_u = torch.zeros_like(u_ts, device=device)
                combined_space_s = torch.zeros_like(s_ts, device=device)
                combined_space_v = torch.zeros_like(v_ts, device=device)
                
            combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
            combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[:n_dims_per_task]
            combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]
        ###################################################################
        
        combined_space_u[:, len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s] = common_space_u
        combined_space_s[len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s] = common_space_s
        combined_space_v[len(config.DATASETS) * n_dims_per_task : len(config.DATASETS) * n_dims_per_task + common_space_index_s, :] = common_space_v
        
        ### Orthogonalize combined_space_u and combined_space_v ###
        u_combined_space_u, s_combined_space_u, v_combined_space_u = torch.linalg.svd(combined_space_u, full_matrices=False)
        u_combined_space_v, s_combined_space_v, v_combined_space_v = torch.linalg.svd(combined_space_v, full_matrices=False)
        combined_space_u = u_combined_space_u @ v_combined_space_u
        combined_space_v = u_combined_space_v @ v_combined_space_v
        ###################################################################
        
        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()
        
        new_vector[key] = torch.linalg.multi_dot(
            (
                combined_space_u,
                torch.diag(combined_space_s),
                combined_space_v,
            )
        )
    
    return new_vector


def wudi_merge(task_vectors, config):
    device = config.device
    print("Computing WUDI...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            new_vector[key] = {}
            orig_vecs = []
            is_linear_vectors = 'attn.in_proj_weight' in key or 'attn.out_proj.weight' in key or 'mlp.c_fc.weight' in key or 'mlp.c_proj.weight' in key
            
            for i, (task_vector, dataset) in enumerate(zip(task_vectors, config.DATASETS)):
                vec = task_vector.vector[key].to(device)
                if (len(task_vector.vector[key].shape) == 2) and ("text_projection" not in key) and is_linear_vectors:
                    orig_vecs.append(vec)
                else:
                    if i == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if (len(task_vector.vector[key].shape) == 2) and ("text_projection" not in key) and is_linear_vectors:
                vecs = torch.stack(orig_vecs, dim=0)
                with torch.enable_grad():
                    new_vector[key] = get_redundant_task_vector(vecs, device, config.method.iter_num)
                
    return new_vector


def get_redundant_task_vector(vectors, device, iter_num=300):
    vectors = vectors.cuda()
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

    return merging_vector.data.detach().to(device)

