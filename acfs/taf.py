import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from torch.distributions.normal import Normal

class TAFAcquisition(AnalyticAcquisitionFunction):
    def __init__(self, model, source_models, rho=0.1, d1=0, d2=0.1, device=None, dtype=None, T=1, multi=False):
        """
        model: current GP model (target)
        source_models: list of old GP models
        rho: bandwidth for Epanechnikov kernel
        d1: number of iterations without decay
        d2: decay rate per iteration for old model weights (0 < d2 <= 1)
        """
        super().__init__(model=model)
        self.model = model
        self.source_models = source_models
        self.M = len(source_models)
        self.rho = float(rho)
        self.d1 = int(d1)
        self.d2 = float(d2)
        self.iteration = 0

        # device / dtype
        self.device = device if device is not None else next(model.parameters()).device
        self.dtype = dtype if dtype is not None else next(model.parameters()).dtype

        self.normal = Normal(torch.tensor(0.0, device=self.device, dtype=self.dtype),
                             torch.tensor(1.0, device=self.device, dtype=self.dtype))

        # dimension of input
        self.D = model.train_inputs[0].shape[-1]
        self.T = T
        self.multi = multi

    def _decay_factor(self):
        k = self.iteration
        if k <= self.d1:
            return 1.0
        elif k <= self.d1 + 1.0 / self.d2:
            return 1.0 - (k - self.d1) * self.d2
        else:
            return 0.0

    def increment_iteration(self):
        self.iteration += 1
    
    def _ranking(self, X_flat, B, q):
        t = X_flat.shape[0]
        chi = []

        for k in range(self.M + 1):
            if k < self.M:
                mu_k = self.source_models[k].posterior(X_flat).mean.squeeze(-1) # shape (t,)
            else:
                mu_k = self.model.posterior(X_flat).mean.squeeze(-1) # shape (t,)

            # Compute pairwise differences: broadcasting (t,1) - (1,t) => (t,t)
            pairwise = mu_k[:, None] - mu_k[None, :]
            # Indicator: 1 if i > j
            comp = (pairwise > 0).float()

            # Flatten into (t^2,) and normalize
            chi_k = comp.reshape(-1) / (t * (t - 1))
            chi.append(chi_k)
        return chi

    # Non-vectorized
    # def _ranking(self, X_flat, B, q):
    #     t = X_flat.shape[0]
    #     chi = [torch.zeros((t ** 2,)) for _ in range(self.M + 1)]
    #     for k in range(self.M + 1):
    #         for i in range(t):
    #             xi = X_flat[i, :].reshape(1, self.D)
    #             mu_k_i, _ = self.source_models[k].posterior(xi).mean.squeeze(-1) if k < self.M \
    #                 else self.model.posterior(xi).mean.squeeze(-1)
    #             for j in range(t):
    #                 xj = X_flat[j, :].reshape(1, self.D)
    #                 mu_k_j, _ = self.source_models[k].posterior(xj).mean.squeeze(-1) if k < self.M \
    #                 else self.model.posterior(xj).mean.squeeze(-1)
    #                 chi[k][j + i * t] = 1 / (t * (t - 1)) if mu_k_i.item() > mu_k_j.item() else 0.0
            # return chi
       

    def _epanechnikov_kernel(self, a: torch.Tensor, b: torch.Tensor):
        # a: ((B*q)**2,), b: ((B*q)**2,) → elementwise kernel
        u = torch.norm(a - b) / self.rho
        out = torch.zeros_like(u)
        mask = u <= 1
        out[mask] = 0.75 * (1 - u[mask]**2)
        return out



    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (..., q, d)
        returns: (..., q) acquisition values
        """
        assert X.shape[-1] == self.D, f"Expected last dim d={self.D}, got {X.shape[-1]}"

        batch_shape = X.shape[:-2]
        q = X.shape[-2]
        B = X.shape[0]
        X_flat = X.view(-1, X.shape[-1])

        # -------------------------
        # 1) Target posterior -> means/stds shaped (B, q)
        # -------------------------
        post_t = self.model.posterior(X_flat)
        means_t = post_t.mean.squeeze(-1)
        stds_t = post_t.variance.sqrt().squeeze(-1).clamp_min(1e-8)


        train_latent_target = self.model.posterior(self.model.train_inputs[0]).mean.squeeze(-1)
        train_latent_target = train_latent_target.detach().requires_grad_(False)
        incumbent_t = train_latent_target.max()
        mask = stds_t > 0
        zs = torch.zeros_like(means_t, dtype=self.dtype, device=self.device)
        zs[mask] = (means_t[mask] - incumbent_t) / stds_t[mask]
        pdf = torch.exp(self.normal.log_prob(zs))
        cdf = self.normal.cdf(zs)
        eis_target = torch.zeros_like(means_t, dtype=self.dtype, device=self.device)
        eis_target[mask] = (means_t[mask] - incumbent_t) * cdf[mask] + stds_t[mask] * pdf[mask]  # (B*q, 1)

        # -------------------------
        # 2) Source predicted improvements -> Is_source shaped (B, q, M)
        # -------------------------
        Is_list = []

        for i, src in enumerate(self.source_models):
            post_s = src.posterior(X_flat)
            means_s = post_s.mean.squeeze(-1)
            train_latent = src.posterior(src.train_inputs[0]).mean.squeeze(-1)
            train_latent = train_latent.detach().requires_grad_(False)
            incumb_s = train_latent.max()  # scalar
            Is = torch.clamp(means_s - incumb_s, min=0.0) #(B*q,)
            Is_list.append(Is)

            
        # stack -> (B*q, M)
        if len(Is_list) > 0:
            Is_source = torch.stack(Is_list, dim=1)
        else:
            # no source models
            Is_source = torch.zeros((B*q, 0), device=self.device, dtype=self.dtype)

        
         # compute weights
        weights = []
        if self.multi:
            chi = self._ranking_multi(X_flat,B,q) # List M x (B*q)**2
        else:
            chi = self._ranking(X_flat,B,q) # List M x (B*q)**2
        for i in range(self.M + 1):
            weights.append((self._epanechnikov_kernel(chi[i], chi[self.M + 1 - 1])).repeat(X_flat.shape[0])) 
        weights_all = torch.stack(weights, dim=-1) # (B*q, M+1)
        weights_old = weights_all[...,:-1]

        # -------------------------
        # 4) Apply decay to old model weights (scalar d(k))
        # -------------------------
        decay = float(self._decay_factor())
        decay_t = torch.tensor(decay, device=self.device, dtype=self.dtype)
        weights_old = weights_old * decay_t  # (B*q, M)


        # normalize per batch
        weights_sum = weights_all.sum(dim=1, keepdim=True)
        weights_all = weights_all / (weights_sum + 1e-12)

        # -------------------------
        # 5) Combine contributions -> taf_vals shape (B, q)
        # -------------------------
        if self.M > 0:
            # broadcast weights_old to (B, q, M) for pointwise multiplication
            w_old_exp = weights_all[..., :-1]  # (B*q, M)
            source_af = (Is_source * w_old_exp).sum(dim=-1)  # (B*q,)
        else:
            source_af = torch.zeros_like(eis_target)

        # add target contribution (broadcast)
        w_target = weights_all[..., -1]  # (B*q,)
        taf_vals = source_af + w_target * eis_target  # (B*q,)

        # taf_vals = eis_target

        # taf_vals = taf_vals.clamp_min(1e-8)
        
        out = taf_vals.view(*batch_shape, q).max(dim=-1).values

        return out
