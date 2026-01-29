import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.monte_carlo import qExpectedImprovement
from torch.distributions.normal import Normal

class TwoStepLookaheadPairwise(AnalyticAcquisitionFunction):
    """
    Two-step lookahead acquisition for PairwiseGP.
    - model: a PairwiseGP trained on (train_X, train_comps)
    - num_fantasies: number of outer fantasies S
    - k: None (use all train points) or int K (use top-K train points to compare)
    - ref_points: pass reference points for fantasizing
    - use_condition_on_observations: if True, call model.condition_on_observations when possible;
        otherwise create a new PairwiseGP on augmented data (slower).
    NOTE: This implementation expects forward(X) with X shape (q, d).
    """
    def __init__(
        self,
        model,                      # PairwiseGP (current model)
        num_fantasies: int = 16,
        fantasy_trials: int = 10,
        k: int = None,  # None -> use all train points
        ref_points = None,
        inner_acq_cls = qExpectedImprovement,
        inner_acq_kwargs = None,
        dtype=torch.double,
        device=None,
        T=1,
        multi=False,
    ):
        super().__init__(model=model)
        self.model = model
        self.num_fantasies = int(num_fantasies)
        self.fantasy_trials = int(fantasy_trials)
        self.k = k
        self.ref_points = ref_points
        self.inner_acq_cls = inner_acq_cls
        self.inner_acq_kwargs = {} if inner_acq_kwargs is None else dict(inner_acq_kwargs)
        self.device = device if device is not None else next(model.parameters()).device
        self.dtype = dtype if dtype is not None else next(model.parameters()).dtype
        self.normal = Normal(loc=0.0, scale=1.0)
        self.T=T
        self.multi=multi
    
    def _select_ref_points(self):
        # Use provided ref_points if given
        if self.ref_points is not None:
            return self.ref_points.to(device=self.device, dtype=self.dtype)

        # Use top-k from training set if requested
        elif self.k is not None:
            train_X = self.model.train_inputs[0].to(self.device).to(self.dtype)
            post = self.model.posterior(train_X)
            mean_util = post.mean.squeeze(-1)
            topk_idx = torch.topk(mean_util, self.k).indices
            return train_X[topk_idx]

        # Default: use all training inputs
        else:
            return self.model.train_inputs[0].to(self.device).to(self.dtype)
        
    def _make_fantasy_comparisons_multi(
        self,
        X_cand: torch.Tensor,
        ref_points: torch.Tensor,
        T: int = 3,
        n_trials: int = 10,
    ) -> [torch.Tensor, torch.Tensor]:
        """
        Generate multiwise (T-ary) comparison indices for fantasies.

        Args:
            X_cand: (q, d) candidate points
            ref_points: (n_ref, d) reference points
            T: number of options per comparison
            n_trials: number of comparisons to generate per fantasy

        Returns:
            comparisons_fantasies: (num_fantasies, n_trials, T) indices into [q + n_ref]
            choices_fantasies: (num_fantasies, n_trials) index in 0..T-1 of the chosen winner
        """
        X_cand = X_cand.to(self.device)
        q, d = X_cand.shape[-2:]
        n_ref = ref_points.shape[0]

        # pool of all points
        X_aug = torch.cat([X_cand, ref_points], dim=0)  # (q + n_ref, d)
        n_total = q + n_ref

        # posterior over all points
        posterior = self.model.posterior(X_aug)
        latent_samples = posterior.rsample(sample_shape=torch.Size([self.num_fantasies]))
        # latent_samples: (num_fantasies, n_total)

        comparisons_fantasies = []
        choices_fantasies = []

        for s in range(self.num_fantasies):
            sample_s = latent_samples[s]  # (n_total,)

            comp_s = []
            choice_s = []

            for _ in range(n_trials):
                # randomly pick T distinct points
                cand = torch.randperm(n_total, device=self.device)[:T]
                utils = sample_s[cand]

                # winner = argmax utility
                winner = torch.argmax(utils).item()

                comp_s.append(cand)
                choice_s.append(winner)

            comparisons_fantasies.append(torch.stack(comp_s))   # (n_trials, T)
            choices_fantasies.append(torch.tensor(choice_s, device=self.device))

        comparisons_fantasies = torch.stack(comparisons_fantasies, dim=0)  # (num_fantasies, n_trials, T)
        choices_fantasies = torch.stack(choices_fantasies, dim=0)          # (num_fantasies, n_trials)

        return comparisons_fantasies, choices_fantasies

    
    def _make_fantasy_comparisons(self, X_cand: torch.Tensor, ref_points: torch.Tensor):
        """
        Generate pairwise comparison indices for multiple fantasies.
        X_cand: (q, d)
        ref_points: (n_ref, d)
        Returns: comparisons_fantasies: (num_fantasies, n_pairs, 2, d)
        """
        X_cand = X_cand.to(self.device)
        q, d = X_cand.shape[-2:]
        n_ref = ref_points.shape[0]

        X_aug = torch.cat([X_cand, ref_points], dim=0)  # (q + n_ref, d)
        # posterior: returns mean, covariance
        posterior = self.model.posterior(X_aug)
        latent_samples = posterior.rsample(sample_shape=torch.Size([self.num_fantasies]))
        # shape: (num_fantasies, q + n_ref)
        comparisons_fantasies = []

        for s in range(self.num_fantasies):
            sample_s = latent_samples[s]  # (q + n_ref,)
            # candidate indices: 0..q-1, reference indices: q..q+n_ref-1
            cand_idx = torch.arange(q, device=self.device)
            ref_idx = torch.arange(q, q + n_ref, device=self.device)

            # Create all candidate vs reference pairs
            cand_idx_grid = cand_idx.unsqueeze(1).repeat(1, n_ref).flatten()
            ref_idx_grid = ref_idx.unsqueeze(0).repeat(q, 1).flatten()

            # Determine winner based on sampled utilities
            sample_c = sample_s[cand_idx_grid]
            sample_r = sample_s[ref_idx_grid]
            winners = sample_c > sample_r

            # Build comparisons tensor: winner > loser
            pairs = torch.stack([cand_idx_grid, ref_idx_grid], dim=-1)  # (n_comparisons, 2)
            winners = winners.view(-1)  # or winners.squeeze(-1)
            pairs_corrected = pairs.clone()
            pairs_corrected[~winners] = pairs_corrected[~winners][:, [1, 0]]  # flip loser rows
            comparisons_fantasies.append(pairs_corrected)

        comparisons_fantasies = torch.stack(comparisons_fantasies, dim=0)  # (num_fantasies, n_comparisons, 2)
        return comparisons_fantasies


    def _sample_fantasy_latents(self, comparisons):
        """
        Sample fantasy outcomes for given pairwise comparisons
        comparisons: (n_pairs, 2, d)
        Returns: 0/1 outcomes of shape (n_fantasies, n_pairs)
        """
        # compute utility difference
        u1 = self.model.posterior(comparisons[:, 0, :]).mean.squeeze(-1)
        u2 = self.model.posterior(comparisons[:, 1, :]).mean.squeeze(-1)
        prob = torch.sigmoid(u1 - u2)  # probability first beats second

        # sample binary outcomes
        return torch.bernoulli(prob.unsqueeze(0).repeat(self.num_fantasies, 1))

    def _evaluate_inner_acq(self, fantasized_model, X_cand: torch.Tensor):
        """Evaluate inner acquisition on fantasized model."""
        try:
            inner_acq = self.inner_acq_cls(model=fantasized_model, **self.inner_acq_kwargs)
        except Exception:
            # fallback: for qEI-like constructors needing best_f
            inner_acq = self.inner_acq_cls(
                model=fantasized_model,
                X_baseline=float(self.model.datapoints)
            )

        # try:
        #     val = inner_acq(X_cand)
        #     print("here1",val.shape)
        # except Exception:
        # manual EI computation
        post = fantasized_model.posterior(X_cand)
        means = post.mean.squeeze(-1)
        stds = post.variance.sqrt().squeeze(-1).clamp(min=1e-8)
        # incumbent = float(fantasized_model.train_targets.max().item())
        train_latent = fantasized_model.posterior(fantasized_model.train_inputs[0]).mean.squeeze(-1)
        incumbent = train_latent.max() 
        mask = stds > 0
        z = torch.zeros_like(means)
        z[mask] = (means[mask] - incumbent) / (stds[mask] + 1e-9)
        pdf = torch.exp(self.normal.log_prob(z))
        cdf = self.normal.cdf(z)
        eis = torch.zeros_like(means)
        eis[mask] = (means[mask] - incumbent) * cdf[mask] + stds[mask] * pdf[mask]
        val = eis

        return val

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of 2-step lookahead acquisition.
        X: (batch_shape, q, d)
        Returns: (batch_shape,) acquisition values
        """
        # print("check-2s---",X.shape)
        X = X.to(self.device)
        batch_shape = X.shape[:-2]
        q, d = X.shape[-2:]
        X_flat = X.view(-1, X.shape[-1])
        ref_points = self._select_ref_points()

        # Generate fantasy comparisons
        if self.multi:
            comparisons_fantasies, choices_fantasies  = self._make_fantasy_comparisons_multi(X_flat, ref_points, T=self.T, n_trials=self.fantasy_trials)
        else:
            comparisons_fantasies = self._make_fantasy_comparisons(X_flat, ref_points)

        # Prepare X_aug for conditioning
        X_aug = torch.cat([X_flat, ref_points], dim=0)  # (q + n_ref, d)

        acq_vals = []
        for s in range(self.num_fantasies):
            if self.multi:
                comps_s = comparisons_fantasies[s]
                ch_s = choices_fantasies[s]

                # Condition MultiwiseGP on fantasy comparisons
                if hasattr(self.model, "condition_on_observations"):
                    fantasized_model = self.model.condition_on_observations(X_aug, comps_s, ch_s)
                else:
                    fantasized_model = self.model
            else:
                comps_s = comparisons_fantasies[s]
                
                # Condition PairwiseGP on fantasy comparisons
                if hasattr(self.model, "condition_on_observations"):
                    fantasized_model = self.model.condition_on_observations(X_aug, comps_s)
                else:
                    fantasized_model = self.model

            val = self._evaluate_inner_acq(fantasized_model, X_flat)

            acq_vals.append(val)

        if len(acq_vals) == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        acq_vals = torch.stack(acq_vals, dim=0)  # (num_fantasies, num_points)

        acq_vals = acq_vals.mean(dim=0)          # average over fantasies -> (num_points,)
        # acq_vals = torch.nan_to_num(acq_vals, nan=-1e8, posinf=-1e8, neginf=-1e8)
        acq_vals = acq_vals.view(*batch_shape, q).mean(dim=-1)
        return acq_vals # now the shape is (RAW_SAMPLES,) #.view(*batch_shape, q)      # restore batch shape (RAW_SAMPLES, q)
