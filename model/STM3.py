import torch
import torch.nn.functional as F
from mamba_ssm.utils.torch import custom_bwd, custom_fwd
from einops import rearrange, repeat
try:
    from causal_conv1d import causal_conv1d_fn
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_cuda = None
from mamba_ssm.ops.triton.layer_norm import _layer_norm_fwd
import selective_scan_cuda

def rms_norm_forward(
    x,
    weight,
    bias,
    eps=1e-6,
    is_rms_norm=True,
):
    # x (b l) d
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y = _layer_norm_fwd(
        x, weight, bias, eps, None, residual_dtype=None, is_rms_norm=is_rms_norm
    )[0]
    # y (b l) d
    return y

class SelectiveScanFn_multiscale(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)

def selective_scan_fn_multiscale(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn_multiscale.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

class Mamba_multiscale(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,
        layer_idx=None,
        scales=[1, 3, 5, 7],
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.scales = scales
        self.num_scales = len(scales)
        self.scale_dim = self.d_inner // self.num_scales

        assert d_model % self.num_scales == 0, f"d_model {d_model} must be divisible by num_scales {self.num_scales}"
        
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.scale_dim,
                out_channels=self.scale_dim,
                kernel_size=scale,
                padding="same",
                bias=conv_bias,
                **factory_kwargs
            ) for scale in scales
        ])
        
        self.dt_embedding = nn.Parameter(torch.randn(self.d_inner))
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def multi_scale_process(self, x):
        B, D, L = x.shape
        x_ms = x.view(B, self.num_scales, self.scale_dim, L)
        
        processed = []
        for q in range(self.num_scales):
            scale_feat = x_ms[:, q]
            conv_feat = self.scale_convs[q](scale_feat)
            processed.append(conv_feat)
        
        return torch.cat(processed, dim=1)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)
        
        x = self.multi_scale_process(x)
        
        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
            
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        
        x_dbl = torch.tanh(self.x_proj(rearrange(x, "b d l -> (b l) d")))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        
        dt = dt + self.dt_embedding.view(1, -1, 1)
        
        y = selective_scan_fn_multiscale(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
            
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EfficientMultiScaleAVWGCN(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        cheb_k,
        embed_dim,
        num_scales,
    ):
        super().__init__()
        self.cheb_k = cheb_k
        self.num_scales = num_scales
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        
        self.q_proj = nn.Linear(dim_out, dim_out)
        self.k_proj = nn.Linear(dim_out, dim_out)
        self.v_proj = nn.Linear(dim_out, dim_out)
        
        self.out_proj = nn.Linear(dim_out, dim_out)
        
        self.norm1 = nn.LayerNorm([dim_out, num_scales])
        self.norm2 = nn.LayerNorm([dim_out, num_scales])

        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.zeros_(self.bias_pool)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, node_embeddings):
        B, N, D, Q = x.shape
        K = self.cheb_k

        spatial_sim = F.relu(torch.mm(node_embeddings, node_embeddings.t()))
        spatial_adj = F.softmax(spatial_sim, dim=1)
        support_set = [torch.eye(N, device=x.device), spatial_adj]
        for k in range(2, K):
            support_set.append(2 * torch.mm(spatial_adj, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set)

        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)

        supports = supports.unsqueeze(1)
        x_g = torch.einsum("kbnm,bmdq->bkndq", supports, x)
        x_g = torch.einsum('bkndq,nkdo->bnoq', x_g, weights)
        conv_results = x_g + bias.unsqueeze(-1)
        
        conv_results = self.norm1(conv_results)

        x_attn = conv_results.transpose(2, 3)
        
        q = self.q_proj(x_attn)
        k = self.k_proj(x_attn)
        v = self.v_proj(x_attn)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_out ** 0.5)
        
        mask = torch.triu(torch.ones(Q, Q, device=x.device))
        attn_scores = attn_scores.masked_fill(mask == 0, -1e4)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attended = torch.matmul(attn_weights, v)
        attended = self.out_proj(attended)
        
        output = conv_results + attended.transpose(2, 3)
        output = self.norm2(output)
        
        return output


class AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MixtureOfExperts(nn.Module):
    def __init__(self, hidden_size, node_emb_dim, num_experts, num_experts_per_tok, 
                 scales=[1,3,5,7], contrast_loss_weight=1.0, 
                 contrastive_temp=0.1,
                 gamma1=0.0, gamma2=0.0):
        super(MixtureOfExperts, self).__init__()
        self.hidden_size = hidden_size
        self.node_emb_dim = node_emb_dim
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.contrast_loss_weight = contrast_loss_weight
        self.contrastive_temp = contrastive_temp
        
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) cannot be greater than num_experts ({self.num_experts})")

        self.gate = nn.Linear(node_emb_dim, num_experts, bias=False)
        
        self.shared_gate = nn.Linear(node_emb_dim, 1, bias=False)

        self.experts = nn.ModuleList(
            [Mamba_multiscale(d_model=hidden_size, d_state=16, d_conv=4, expand=2, scales=scales) for _ in range(num_experts)]
        )
        self.shared_expert = Mamba_multiscale(d_model=hidden_size, d_state=16, d_conv=4, expand=2, scales=scales)

        self.gamma1 = gamma1
        self.gamma2 = gamma2


    def _compute_contrast_loss(self, expert_outputs, expert_indices):
        batch_size, num_nodes, num_experts, temporal_dim, sub_hidden_dim, scale_dim = expert_outputs.shape
        
        expert_reps = expert_outputs.mean(dim=3)
        expert_reps = F.normalize(expert_reps, p=2, dim=3)
        
        reps = expert_reps.permute(0, 1, 2, 4, 3).reshape(batch_size * num_nodes, num_experts, scale_dim, sub_hidden_dim)
        
        total_loss = 0.0
        valid_experts = 0
        temp = self.contrastive_temp
        num_negatives = min(10, batch_size * num_nodes - 1)
        
        expert_indices = expert_indices.reshape(-1)
        
        p_indices = torch.arange(scale_dim, device=expert_outputs.device)
        q_indices = torch.arange(scale_dim, device=expert_outputs.device)
        p, q = torch.meshgrid(p_indices, q_indices, indexing='ij')
        
        pos_mask = p > q
        neg_mask = p <= q
        
        weights = torch.zeros_like(p, dtype=torch.float32)
        
        weights[pos_mask] = torch.pow((p[pos_mask] - q[pos_mask] + 1).float(), -self.gamma1)
        
        weights[neg_mask] = torch.pow((q[neg_mask] - p[neg_mask] + 1).float(), self.gamma2)
        
        for e in range(num_experts):
            mask = (expert_indices == e)
            idxs = torch.where(mask)[0]
            if idxs.numel() == 0:
                continue
            valid_experts += 1
            
            samples = reps[idxs, e]
            N = samples.shape[0]
            
            other_idxs = torch.where(expert_indices != e)[0]
            
            sim_matrix = torch.matmul(samples, samples.transpose(1,2)) / temp
            
            sim_matrix = sim_matrix * weights.unsqueeze(0)
            upper_tri = torch.triu(torch.ones(scale_dim, scale_dim, device=samples.device), diagonal=1).bool()
            sim_matrix = sim_matrix * upper_tri.unsqueeze(0)
            
            if len(other_idxs) > 0:
                neg_idx = torch.randint(0, len(other_idxs), (N, num_negatives))
                neg_global_idx = other_idxs[neg_idx]
                neg_expert = torch.randint(0, num_experts-1, (N, num_negatives))
                neg_expert[neg_expert >= e] += 1
                neg_scale = torch.randint(0, scale_dim, (N, num_negatives))
                neg_features = reps[neg_global_idx, neg_expert, neg_scale]
                neg_sim = torch.matmul(samples, neg_features.transpose(1,2)) / temp
            else:
                neg_sim = torch.zeros(N, scale_dim, num_negatives, device=samples.device) - 10.0
            
            full_sim = torch.cat([sim_matrix, neg_sim], dim=2)
            
            pos_mask = torch.zeros(N, scale_dim, scale_dim, dtype=torch.bool, device=samples.device)
            rows, cols = torch.triu_indices(scale_dim, scale_dim, offset=1)
            pos_mask[:, rows, cols] = True
            pos_mask[:, 1:, :-1] = True
            
            full_pos_mask = torch.cat([
                pos_mask,
                torch.zeros(N, scale_dim, num_negatives, dtype=torch.bool, device=samples.device)
            ], dim=2)
            
            log_p = F.log_softmax(full_sim.reshape(N*scale_dim, -1), dim=1)
            pos_logprob = (log_p * full_pos_mask.reshape(N*scale_dim, -1)).sum() / (full_pos_mask.sum() + 1e-8)
            total_loss += -pos_logprob

        return total_loss / valid_experts if valid_experts > 0 else torch.tensor(0.0, device=expert_outputs.device)

    def forward(self, hidden_states: torch.Tensor, node_embeddings: torch.Tensor):
        batch_size, num_nodes, temporal_dim, sub_hidden_dim, scale_dim = hidden_states.shape

        flat_node_emb = node_embeddings
        if batch_size > 1:
            flat_node_emb = flat_node_emb.unsqueeze(0).expand(batch_size, -1, -1)
        flat_node_emb = flat_node_emb.reshape(batch_size * num_nodes, -1)
        flat_hidden = hidden_states.reshape(batch_size * num_nodes, temporal_dim, -1)
        
        router_logits = self.gate(flat_node_emb)
        routing_weights = F.softmax(router_logits + torch.rand_like(router_logits) * 0.01, dim=1)
        routing_weights, expert_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-8)

        final_outputs = torch.zeros(
            batch_size * num_nodes, 
            temporal_dim, 
            sub_hidden_dim * scale_dim,
            device=flat_hidden.device,
            dtype=flat_hidden.dtype
        )
        
        if self.training and self.contrast_loss_weight > 0:
            contrast_outputs = torch.zeros(
                batch_size * num_nodes,
                self.num_experts,
                temporal_dim,
                sub_hidden_dim,
                scale_dim,
                device=flat_hidden.device,
                dtype=flat_hidden.dtype
            )
        else:
            contrast_outputs = None

        for i, expert in enumerate(self.experts):
            mask = (expert_indices == i).any(dim=1)
            if not mask.any():
                continue
                
            expert_weight_mask = (expert_indices[mask] == i)
            expert_weights = routing_weights[mask].gather(1, expert_weight_mask.nonzero()[:, 1:])

            expert_out = expert(flat_hidden[mask])
            final_outputs[mask] += expert_out * expert_weights.unsqueeze(-1)

            if contrast_outputs is not None:
                contrast_outputs[mask, i] = expert_out.view(-1, temporal_dim, sub_hidden_dim, scale_dim)

        final_outputs = final_outputs.view(batch_size * num_nodes, temporal_dim, sub_hidden_dim, scale_dim)
        final_outputs = rearrange(final_outputs, '(b n) t s d -> b n t s d', b=batch_size, n=num_nodes)

        if hasattr(self, 'shared_expert'):
            shared_weight = torch.sigmoid(self.shared_gate(flat_node_emb))
            shared_weight = shared_weight.view(batch_size, num_nodes, 1, 1, 1)
            
            shared_output = self.shared_expert(flat_hidden)
            shared_output = shared_output.view(batch_size * num_nodes, temporal_dim, sub_hidden_dim, scale_dim)
            shared_output = rearrange(shared_output, '(b n) t s d -> b n t s d', b=batch_size, n=num_nodes)
            
            final_outputs = final_outputs + shared_weight * shared_output

        if self.training and contrast_outputs is not None:
            contrast_outputs = rearrange(contrast_outputs, '(b n) e t s d -> b n e t s d', b=batch_size, n=num_nodes)
            contrast_loss = self._compute_contrast_loss(contrast_outputs, expert_indices)
            total_aux_loss = self.contrast_loss_weight * contrast_loss
            
            final_outputs = AddAuxiliaryLoss.apply(final_outputs, total_aux_loss)

        return final_outputs, router_logits


class STM3Layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.hidden_dim = args.feature_hidden_dim * len(args.scales)
        
        self.dgcn = EfficientMultiScaleAVWGCN(
            dim_in=args.feature_hidden_dim,
            dim_out=args.feature_hidden_dim,
            cheb_k=3, 
            embed_dim=args.node_emb_dim,
            num_scales=len(args.scales),
        )
        self.moe = MixtureOfExperts(
            hidden_size=self.hidden_dim,
            node_emb_dim=args.node_emb_dim,
            num_experts=args.num_experts,
            num_experts_per_tok=1,
            scales=args.scales,
            contrast_loss_weight=args.contrast_loss_weight,
            contrastive_temp=args.contrastive_temp,
            gamma1=args.gamma1,
            gamma2=args.gamma2,
        )

    def forward(self, x, node_embeddings):
        B, L, N, D, Q = x.shape
        batch_size = x.shape[0]
        
        x = rearrange(x, 'b l n d q -> (b l) n d q', 
                    b=batch_size, l=self.args.lag, n=self.args.num_nodes, d=self.args.feature_hidden_dim, q=len(self.args.scales))
        dgcn_out = self.dgcn(x, node_embeddings)
        x = x + dgcn_out
        
        x = rearrange(x, '(b l) n d q -> b n l d q', 
                    b=batch_size, l=self.args.lag, n=self.args.num_nodes, d=self.args.feature_hidden_dim, q=len(self.args.scales))
        
        moe_output, _ = self.moe(x, node_embeddings)
        x = x + moe_output
        x = rearrange(x, 'b n l d q -> b l n d q', 
                    b=batch_size, l=self.args.lag, n=self.args.num_nodes, d=self.args.feature_hidden_dim, q=len(self.args.scales))

        return x


class STM3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.hidden_dim = args.feature_hidden_dim * len(args.scales)

        self.multiscale_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=args.feature_hidden_dim,
                out_channels=args.feature_hidden_dim,
                kernel_size=scale,
                padding="same",
            ) for scale in args.scales
        ])
        
        self.feature_transform = nn.Sequential(
            nn.Linear(self.args.input_dim, self.args.feature_hidden_dim)
        )

        self.node_embeddings = nn.Parameter(
            torch.randn(self.args.num_nodes, self.args.node_emb_dim), 
            requires_grad=True
        )
        nn.init.xavier_uniform_(self.node_embeddings)
        
        self.layers = nn.ModuleList([
            STM3Layer(self.args) for _ in range(self.args.num_layers)
        ])

        self.time_distributed = nn.Sequential(
            nn.Linear(
                self.hidden_dim * self.args.lag, 
                self.args.output_dim * self.args.horizon
            )
        )

    def extract_multiscale_features(self, x):
        batch_size, num_nodes, seq_len, hidden_dim = x.shape
        x = rearrange(x, 'b n l d -> (b n) d l', 
                    b=batch_size, n=num_nodes, l=seq_len, d=hidden_dim)
        scale_features = []
        for conv in self.multiscale_convs:
            conv_out = conv(x)
            scale_features.append(conv_out)
        x = torch.stack(scale_features, dim=-1)
        x = rearrange(x, '(b n) d l q -> b n l d q', 
                    b=batch_size, n=num_nodes, l=seq_len, d=hidden_dim)
        return x

    def forward(self, x):
        x = x[:, :, :, :self.args.input_dim]
        batch_size = x.shape[0]
        x = self.feature_transform(x)
        x = self.extract_multiscale_features(x)
        for layer in self.layers:
            x = layer(x, self.node_embeddings)
        x = rearrange(x, 'b l n d q -> b n (l d q)', 
                     b=batch_size, n=self.args.num_nodes, l=self.args.lag, d=self.args.feature_hidden_dim, q=len(self.args.scales))
        out = self.time_distributed(x)
        out = rearrange(out, 'b n (l d) -> b l n d', 
                       b=batch_size, n=self.args.num_nodes, l=self.args.horizon, d=self.args.output_dim)
        return out

