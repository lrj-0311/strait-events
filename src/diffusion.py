import torch
import torch.nn as nn
import torch.nn.functional as F


class ScenarioDiffusionGenerator(nn.Module):
    """Lightweight latent diffusion module for future-scenario sampling.

    The module denoises entity-state embeddings rather than generating raw
    events directly. MUSE-ED can then decode multiple sampled entity states to
    estimate prediction variance, entropy, and unstable future windows.
    """

    def __init__(self, h_dim, num_steps=8, noise_scale=0.15, dropout=0.1, condition_dim=0):
        super().__init__()
        self.h_dim = h_dim
        self.num_steps = max(1, int(num_steps))
        self.noise_scale = float(noise_scale)
        self.condition_dim = condition_dim

        betas = torch.linspace(1e-4, 2e-2, self.num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

        self.step_encoder = nn.Embedding(self.num_steps, h_dim)
        if condition_dim > 0:
            self.condition_proj = nn.Linear(condition_dim, h_dim)
            denoiser_input_dim = h_dim * 3
        else:
            self.condition_proj = None
            denoiser_input_dim = h_dim * 2

        self.denoiser = nn.Sequential(
            nn.Linear(denoiser_input_dim, h_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim * 2, h_dim),
        )

    def _broadcast_condition(self, condition_state, num_nodes):
        if self.condition_proj is None or condition_state is None:
            return None
        if condition_state.dim() == 1:
            condition_state = condition_state.unsqueeze(0)
        condition_state = self.condition_proj(condition_state.float())
        if condition_state.shape[0] == 1:
            condition_state = condition_state.expand(num_nodes, -1)
        return condition_state

    def q_sample(self, clean_state, step_ids, noise=None):
        if noise is None:
            noise = torch.randn_like(clean_state)
        sqrt_alpha = self.sqrt_alpha_bars[step_ids].view(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[step_ids].view(-1, 1)
        noisy_state = sqrt_alpha * clean_state + sqrt_one_minus * self.noise_scale * noise
        return F.normalize(noisy_state, dim=-1)

    def denoise(self, noisy_state, step_ids, condition_state=None):
        step_state = self.step_encoder(step_ids)
        pieces = [noisy_state, step_state]
        condition_state = self._broadcast_condition(condition_state, noisy_state.shape[0])
        if condition_state is not None:
            pieces.append(condition_state)
        clean_hat = self.denoiser(torch.cat(pieces, dim=-1))
        return F.normalize(clean_hat, dim=-1)

    def reconstruction_loss(self, clean_state, condition_state=None):
        step_ids = torch.randint(
            low=0,
            high=self.num_steps,
            size=(clean_state.shape[0],),
            device=clean_state.device,
        )
        noisy_state = self.q_sample(clean_state.detach(), step_ids)
        clean_hat = self.denoise(noisy_state, step_ids, condition_state)
        return F.mse_loss(clean_hat, clean_state.detach())

    def sample(self, base_state, condition_state=None, num_samples=4):
        samples = []
        num_samples = max(1, int(num_samples))
        for _ in range(num_samples):
            state = F.normalize(base_state + self.noise_scale * torch.randn_like(base_state), dim=-1)
            for step in reversed(range(self.num_steps)):
                step_ids = torch.full(
                    (base_state.shape[0],),
                    step,
                    dtype=torch.long,
                    device=base_state.device,
                )
                clean_hat = self.denoise(state, step_ids, condition_state)
                if step > 0:
                    jitter_scale = self.noise_scale * 0.1 * (step / self.num_steps)
                    state = F.normalize(0.75 * clean_hat + 0.25 * state + jitter_scale * torch.randn_like(state),
                                        dim=-1)
                else:
                    state = clean_hat
            samples.append(state)
        return torch.stack(samples, dim=0)
