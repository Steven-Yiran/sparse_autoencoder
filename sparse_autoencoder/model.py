from typing import Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

def LayerNorm(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std

class Autoencoder(nn.Module):
    """Sparse Autoencoder

    Implements the autoencoder defined in
    https://transformer-circuits.pub/2023/monosemantic-features/index.html
    Specifically, given encoder wrights W_enc, decoder weights W_dec with 
    column of unit norm, and biases b_enc, b_dec, the forward pass is given by:
        Encode: latent = ReLu(W_enc @ (x - b_dec) + b_enc)
        Decode: recons = W_dec @ f + b_dec
    """

    def __init__(
        self,
        n_latents: int,
        n_inputs: int,
        activation: Callable = nn.Relu(),
        tied: bool = False,
        normalize: bool = False,
    ) -> None:
        """
        Args:
            n_latents: Number of latent units
            n_inputs: Number of input units
            activation: Activation function
            tied: Whether to tie the weights of the encoder and decoder
            normalize: Whether to normalize the weights
        """
        super().__init__()

        self.b_dec = nn.Parameter(torch.zeros(n_inputs))
        self.encode = nn.Linear(n_inputs, n_latents, bias=False)
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.activation = activation
        self.decoder = nn.Linear(n_latents, n_inputs, bias=False)
        self.normalize = normalize

        self.stats_last_nonzero: torch.Tensor
        self.latents_activation_frequency: torch.Tensor
        self.latents_mean_square: torch.Tensor
        self.register_buffer("stats_last_nonzero", torch.zeros(n_latents, dtype=torch.long))
        self.register_buffer(
            "latents_activation_frequency", torch.zeros(n_latents, dtype=torch.float)
        )
        self.register_buffer("latents_mean_square", torch.zeros(n_latents, dtype=torch.float))

    def encode_pre_act(self, x: torch.Tensor, latent_slice: slice = slice(None)) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, n_inputs)
            latent_slice: Slice of the latent units to compute
                Example: slice(0, 10) computes the first 10 latent units
        Returns:
            latents_pre_act: Pre-activation of the latent representation tensor (batch_size, n_latents)
        """
        x = x - self.b_dec
        latents_pre_act = F.linear(
            x, self.encoder.weight[latent_slice], self.b_enc[latent_slice]
        )
        return latents_pre_act

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.normalize:
            return x, dict()
        x, mu, std = LayerNorm(x)
        return x, dict(mu=mu, std=std)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Args:
            x: Input tensor of shape (batch_size, n_inputs)
        Returns:
            latents: Latent representation tensor (batch_size, n_latents)
            info: Dictionary of information needed for decoding
        """
        x, info = self.preprocess(x)
        latents = self.activation(self.encode_pre_act(x))
        return latents, info

    def decode(self, latents: torch.Tensor, info: dict[str, Any]) -> torch.Tensor:
        """
        Args:
            latents: Latent representation tensor of shape (batch_size, n_latents)
            info: Dictionary of information needed for decoding
        Returns:
            recons: Reconstructed input tensor of shape (batch_size, n_inputs)
        """
        recons = self.decoder(latents) + self.b_dec
        if self.normalize:
            assert info is not None
            recons = recons * info["std"] + info["mu"]
        return recons

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, n_inputs)
        Returns:
            latents_pre_act: Pre-activation of the latent representation tensor (batch_size, n_latents)
            latents: Latent representation tensor (batch_size, n_latents)
            recons: Reconstructed input tensor (batch_size, n_inputs)
        """
        x, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x)
        latents = self.activation(latents_pre_act)
        recons = self.decode(latents, info)

        return latents_pre_act, latents, recons
    
    @classmethod
    def from_state_dict(
        cls, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> "Autoencoder":
        n_latents, d_model = state_dict["encoder.weight"].shape

        activation_class_name = state_dict.pop("activation", "ReLU")
        activation_class = ACTIVATION_CLASSES.get(activation_class_name, nn.ReLU)
        normalize = activation_class_name == "TopK"
        activation_state_dict = state_dict.pop("activation_state_dict", {})
        if hasattr(activation_class, "from_state_dict"):
            activation = activation_class.from_state_dict(
                activation_state_dict, strict=strict
            )
        else:
            activation = activation_class()
            if hasattr(activation, "load_state_dict"):
                activation.load_state_dict(activation_state_dict, strict=strict)

        autoencoder = cls(n_latents, d_model, activation=activation, normalize=normalize)
        # Load remaining state dict
        autoencoder.load_state_dict(state_dict, strict=strict)
        return autoencoder
    
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        sd[prefix + "activation"] = self.activation.__class__.__name__
        if hasattr(self.activation, "state_dict"):
            sd[prefix + "activation_state_dict"] = self.activation.state_dict()
        return sd
    

ACTIVATION_CLASSES = {
    "ReLU": nn.ReLU,
    #"TopK": TopK,
    "Identity": nn.Identity,
}