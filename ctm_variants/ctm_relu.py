import torch
import torch.nn as nn
import numpy as np
import math
from huggingface_hub import PyTorchModelHubMixin

from continuous_thought_machines.models.modules import (
    ParityBackbone,
    SynapseUNET,
    Squeeze,
    LearnableFourierPositionalEncoding,
    MultiLearnableFourierPositionalEncoding,
    CustomRotationalEmbedding,
    CustomRotationalEmbedding1D,
    ShallowWide,
)
from continuous_thought_machines.models.resnet import prepare_resnet_backbone
from continuous_thought_machines.models.utils import compute_normalized_entropy

from continuous_thought_machines.models.constants import (
    VALID_NEURON_SELECT_TYPES,
    VALID_BACKBONE_TYPES,
    VALID_POSITIONAL_EMBEDDING_TYPES,
)


class ContinuousThoughtMachineReLU(nn.Module, PyTorchModelHubMixin):
    """
    CTM variant WITHOUT neuron-level models.

    - No per-neuron internal learning
    - No neuron memory traces
    - Shared MLP + ReLU after synapses
    - Depth replaces neuron-internal capacity
    """

    def __init__(
        self,
        iterations,
        d_model,
        d_input,
        heads,
        n_synch_out,
        n_synch_action,
        synapse_depth,
        memory_length,              # kept for API compatibility, unused
        deep_nlms,                  # ignored
        memory_hidden_dims,         # ignored
        do_layernorm_nlm,           # ignored
        backbone_type,
        positional_embedding_type,
        out_dims,
        prediction_reshaper=[-1],
        dropout=0.0,
        neuron_select_type="random-pairing",
        n_random_pairing_self=0,
    ):
        super().__init__()

        # --------------------
        # Core params
        # --------------------
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.out_dims = out_dims
        self.prediction_reshaper = prediction_reshaper
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.backbone_type = backbone_type
        self.positional_embedding_type = positional_embedding_type
        self.neuron_select_type = neuron_select_type

        self.verify_args()

        # --------------------
        # Input / backbone
        # --------------------
        self.set_initial_rgb()
        self.set_backbone()
        d_backbone = self.get_d_backbone()
        self.positional_embedding = self.get_positional_embedding(d_backbone)

        self.kv_proj = nn.Sequential(
            nn.LazyLinear(d_input),
            nn.LayerNorm(d_input),
        )
        self.q_proj = nn.LazyLinear(d_input)
        self.attention = nn.MultiheadAttention(
            d_input, heads, dropout=dropout, batch_first=True
        )

        # --------------------
        # Synapses (unchanged)
        # --------------------
        self.synapses = self.get_synapses(synapse_depth, d_model, dropout)

        # --------------------
        # NEW: shared post-synapse MLP
        # --------------------
        self.post_synapse_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        # --------------------
        # Initial state
        # --------------------
        self.register_parameter(
            "start_activated_state",
            nn.Parameter(torch.zeros(d_model).uniform_(-0.1, 0.1)),
        )

        # --------------------
        # Synchronisation
        # --------------------
        self.neuron_select_type_out, self.neuron_select_type_action = self.get_neuron_select_type()
        self.synch_representation_size_out = self.calculate_synch_representation_size(n_synch_out)
        self.synch_representation_size_action = self.calculate_synch_representation_size(n_synch_action)

        if self.synch_representation_size_action > 0:
            self.set_synchronisation_parameters("action", n_synch_action, n_random_pairing_self)
        self.set_synchronisation_parameters("out", n_synch_out, n_random_pairing_self)

        # --------------------
        # Output
        # --------------------
        self.output_projector = nn.Sequential(nn.LazyLinear(out_dims))

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, x, track=False):
        print("Entered forward")
        B = x.size(0)
        device = x.device

        kv = self.compute_features(x)

        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)

        predictions = torch.empty(B, self.out_dims, self.iterations, device=device)
        certainties = torch.empty(B, 2, self.iterations, device=device)

        decay_alpha_out, decay_beta_out = None, None
        decay_alpha_action, decay_beta_action = None, None

        self.decay_params_action.data.clamp_(0, 15)
        self.decay_params_out.data.clamp_(0, 15)

        r_action = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        for t in range(self.iterations):
            # --- Action sync
            sync_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                activated_state, decay_alpha_action, decay_beta_action, r_action, "action"
            )

            # --- Attention
            q = self.q_proj(sync_action).unsqueeze(1)
            attn_out, _ = self.attention(q, kv, kv, need_weights=False)
            attn_out = attn_out.squeeze(1)

            # --- Synapses
            syn_in = torch.cat([attn_out, activated_state], dim=-1)
            state = self.synapses(syn_in)

            # --- Shared nonlinearity (key change)
            activated_state = self.post_synapse_mlp(state)

            # --- Output sync
            sync_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, "out"
            )

            pred = self.output_projector(sync_out)
            cert = self.compute_certainty(pred)

            predictions[..., t] = pred
            certainties[..., t] = cert

        return predictions, certainties, sync_out

    # ======================================================
    # Utility + setup (mostly unchanged)
    # ======================================================

    def compute_features(self, x):
        rgb = self.initial_rgb(x)
        feats = self.backbone(rgb)
        pos = self.positional_embedding(feats)
        feats = (feats + pos).flatten(2).transpose(1, 2)
        return self.kv_proj(feats)

    def compute_certainty(self, pred):
        B = pred.size(0)
        reshaped = pred.reshape([B] + self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped)
        return torch.stack((ne, 1 - ne), -1)

    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        if synch_type == "action":
            left = self.action_neuron_indices_left
            right = self.action_neuron_indices_right
        else:
            left = self.out_neuron_indices_left
            right = self.out_neuron_indices_right

        if self.neuron_select_type == "random-pairing":
            prod = activated_state[:, left] * activated_state[:, right]
        else:
            sel_l = activated_state[:, left]
            sel_r = activated_state[:, right]
            outer = sel_l.unsqueeze(2) * sel_r.unsqueeze(1)
            i, j = torch.triu_indices(sel_l.size(1), sel_l.size(1))
            prod = outer[:, i, j]

        if decay_alpha is None:
            decay_alpha = prod
            decay_beta = torch.ones_like(prod)
        else:
            decay_alpha = r * decay_alpha + prod
            decay_beta = r * decay_beta + 1

        return decay_alpha / torch.sqrt(decay_beta), decay_alpha, decay_beta

    # --------------------
    # Setup helpers
    # --------------------

    def set_initial_rgb(self):
        if "resnet" in self.backbone_type:
            self.initial_rgb = nn.LazyConv2d(3, 1, 1)
        else:
            self.initial_rgb = nn.Identity()

    def get_d_backbone(self):
        if self.backbone_type == "shallow-wide":
            return 2048
        elif self.backbone_type == "parity_backbone":
            return self.d_input
        elif "resnet" in self.backbone_type:
            return 512
        elif self.backbone_type == "none":
            return None
        else:
            raise ValueError

    def set_backbone(self):
        if self.backbone_type == "shallow-wide":
            self.backbone = ShallowWide()
        elif self.backbone_type == "parity_backbone":
            self.backbone = ParityBackbone(2, self.d_input)
        elif "resnet" in self.backbone_type:
            self.backbone = prepare_resnet_backbone(self.backbone_type)
        elif self.backbone_type == "none":
            self.backbone = nn.Identity()

    def get_positional_embedding(self, d):
        if self.positional_embedding_type == "none":
            return lambda x: 0
        elif self.positional_embedding_type == "learnable-fourier":
            return LearnableFourierPositionalEncoding(d)
        elif self.positional_embedding_type == "multi-learnable-fourier":
            return MultiLearnableFourierPositionalEncoding(d)
        elif self.positional_embedding_type == "custom-rotational":
            return CustomRotationalEmbedding(d)
        elif self.positional_embedding_type == "custom-rotational-1d":
            return CustomRotationalEmbedding1D(d)
        else:
            raise ValueError

    def get_synapses(self, depth, d_model, dropout):
        if depth == 1:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model),
            )
        return SynapseUNET(d_model, depth, 16, dropout)

    def set_synchronisation_parameters(self, kind, n, n_self):
        left, right = self.initialize_left_right_neurons(kind, self.d_model, n, n_self)
        size = self.calculate_synch_representation_size(n)
        self.register_buffer(f"{kind}_neuron_indices_left", left)
        self.register_buffer(f"{kind}_neuron_indices_right", right)
        self.register_parameter(f"decay_params_{kind}", nn.Parameter(torch.zeros(size)))

    def initialize_left_right_neurons(self, kind, d, n, n_self):
        idx = np.random.choice(d, size=n, replace=False)
        left = torch.tensor(idx)
        right = torch.tensor(idx)
        return left, right

    def get_neuron_select_type(self):
        return self.neuron_select_type, self.neuron_select_type

    def calculate_synch_representation_size(self, n):
        return n if self.neuron_select_type == "random-pairing" else (n * (n + 1)) // 2

    def verify_args(self):
        assert self.neuron_select_type in VALID_NEURON_SELECT_TYPES
        assert self.backbone_type in VALID_BACKBONE_TYPES + ["none"]
        assert self.positional_embedding_type in VALID_POSITIONAL_EMBEDDING_TYPES + ["none"]