from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, final

import numpy as np

from slicegpt.model_adapter import SlicingConfig


class SlicingScheduler(ABC):
    """
    Provides the slicing dimensions for each component of the model.

    Can be data dependent (e.g. computing dimensions based on the results of rotation),
    use a parametric model (e.g. spline), or can be based on a provided config.

    Saves the slicing dimensions obtained through this scheduler into a running config that can be serialised
    and later used to re-slice the loaded model.
    """

    def __init__(self, *, do_slice_head: bool = False):
        self.slicing_conf: SlicingConfig = SlicingConfig()
        self.slicing_conf.do_slice_head = do_slice_head

        # eigenvalues obtained from PCA
        self.embedding_eigenvalues: list[float] = []
        self.attention_eigenvalues: dict[int, list[float]] = {}
        self.mlp_eigenvalues: dict[int, list[float]] = {}

    @property
    def do_slice_head(self) -> bool:
        """Return whether to slice the head."""
        return self.slicing_conf.do_slice_head

    @property
    def hidden_size(self) -> int:
        """Return the hidden size."""
        return self.slicing_conf.hidden_size

    @property
    def layers_num(self) -> int:
        """Return the number of layers."""
        return self.slicing_conf.layers_num

    @property
    def parallel_blocks(self) -> bool:
        """Return whether working with a parallel blocks models."""
        return self.slicing_conf.parallel_blocks

    def setup(self, *, hidden_size: int, layers_num: int, parallel_blocks: bool) -> None:
        """Set up the slicing scheduler with the given model parameters."""
        self.slicing_conf.hidden_size = hidden_size
        self.slicing_conf.layers_num = layers_num
        self.slicing_conf.parallel_blocks = parallel_blocks

    @final
    def get_embedding_dimensions(self) -> dict[int, int]:
        """Return the input embedding dimensions."""
        if self.slicing_conf.embedding_dimensions:
            return self.slicing_conf.embedding_dimensions

        val = self._get_embedding_dimensions()
        self.slicing_conf.embedding_dimensions = val
        return val

    @abstractmethod
    def _get_embedding_dimensions(self) -> dict[int, int]:
        raise NotImplementedError

    @final
    def get_attention_input_dimension(self, idx: int) -> int:
        """Return the attention input dimension for the specified layer index."""
        if idx in self.slicing_conf.attention_input_dimensions:
            return self.slicing_conf.attention_input_dimensions[idx]

        val = self._get_attention_input_dimension(idx)
        self.slicing_conf.attention_input_dimensions[idx] = val
        return val

    @abstractmethod
    def _get_attention_input_dimension(self, idx: int) -> int:
        raise NotImplementedError

    @final
    def get_mlp_output_dimension(self, idx: int) -> int:
        """Return the mlp output dimension for the specified layer index."""
        if idx in self.slicing_conf.mlp_output_dimensions:
            return self.slicing_conf.mlp_output_dimensions[idx]

        use_head_dim = idx == self.layers_num - 1
        val = self._get_mlp_output_dimension(idx) if not use_head_dim else self.get_head_dimension()
        self.slicing_conf.mlp_output_dimensions[idx] = val
        return val

    @abstractmethod
    def _get_mlp_output_dimension(self, idx: int) -> int:
        raise NotImplementedError

    @final
    def get_attention_output_dimension(self, idx, match_head_dim: bool | None = None) -> int:
        """Return the attention output dimension for the specified layer index."""
        if self.parallel_blocks:
            return self.get_mlp_output_dimension(idx)

        if idx in self.slicing_conf.attention_output_dimensions:
            return self.slicing_conf.attention_output_dimensions[idx]

        use_head_dim = idx == self.layers_num - 1 and match_head_dim
        val = self._get_attention_output_dimension(idx) if not use_head_dim else self.get_head_dimension()
        self.slicing_conf.attention_output_dimensions[idx] = val
        return val

    @abstractmethod
    def _get_attention_output_dimension(self, idx: int) -> int:
        raise NotImplementedError

    @final
    def get_mlp_input_dimension(self, idx: int) -> int:
        """Return the mlp input dimension for the specified layer index."""
        if self.parallel_blocks:
            return self.get_attention_input_dimension(idx)

        if idx in self.slicing_conf.mlp_input_dimensions:
            return self.slicing_conf.mlp_input_dimensions[idx]

        val = self._get_mlp_input_dimension(idx)
        self.slicing_conf.mlp_input_dimensions[idx] = val
        return val

    @abstractmethod
    def _get_mlp_input_dimension(self, idx: int) -> int:
        raise NotImplementedError

    @final
    def get_head_dimension(self) -> int:
        """Return the LM head dimension."""
        if self.slicing_conf.head_dimension is not None:
            return self.slicing_conf.head_dimension

        val = self._get_head_dimension() if self.slicing_conf.do_slice_head else self.hidden_size
        self.slicing_conf.head_dimension = val
        return val

    @abstractmethod
    def _get_head_dimension(self) -> int:
        raise NotImplementedError

    def set_embedding_eigenvalues(self, eigenvalues: list[float]) -> None:
        """Set the eigenvalues of the embeddings PCA."""
        self.embedding_eigenvalues = eigenvalues

    def set_attention_eigenvalues(self, idx: int, eigenvalues: list[float]) -> None:
        """Set the eigenvalues of the attention layer PCA."""
        self.attention_eigenvalues[idx] = eigenvalues

    def set_mlp_eigenvalues(self, idx: int, eigenvalues: list[float]) -> None:
        """Set the eigenvalues of the MLP layer PCA."""
        self.mlp_eigenvalues[idx] = eigenvalues


class ConfigSlicingScheduler(SlicingScheduler):
    """Slicing scheduler that returns the dimensions specified in the config."""

    def __init__(self, config: SlicingConfig):
        super().__init__()
        self.slicing_conf = config

    def _get_embedding_dimensions(self) -> dict[int, int]:
        return self.slicing_conf.embedding_dimensions

    def _get_attention_input_dimension(self, idx: int) -> int:
        return self.slicing_conf.attention_input_dimensions[idx]

    def _get_mlp_output_dimension(self, idx: int) -> int:
        return self.slicing_conf.mlp_output_dimensions[idx]

    def _get_attention_output_dimension(self, idx: int) -> int:
        return self.slicing_conf.attention_output_dimensions[idx]

    def _get_mlp_input_dimension(self, idx: int) -> int:
        return self.slicing_conf.mlp_input_dimensions[idx]

    def _get_head_dimension(self) -> int:
        return self.slicing_conf.head_dimension


class ConstSlicingScheduler(SlicingScheduler):
    """Slicing scheduler that returns the same dimension for all components."""

    def __init__(self, dimension: int, *, do_slice_head: bool = False):
        super().__init__(do_slice_head=do_slice_head)
        self.dimension: int = dimension

    def _get_embedding_dimensions(self) -> dict[int, int]:
        return defaultdict(lambda: self.dimension)

    def _get_attention_input_dimension(self, idx: int) -> int:
        return self.dimension

    def _get_mlp_output_dimension(self, idx: int) -> int:
        return self.dimension

    def _get_attention_output_dimension(self, idx: int) -> int:
        return self.dimension

    def _get_mlp_input_dimension(self, idx: int) -> int:
        return self.dimension

    def _get_head_dimension(self) -> int:
        return self.dimension


class ForwardSlicingScheduler(SlicingScheduler, ABC):
    """
    An abstract scheduler that enforces dimension consistency across layers.
    Applicable only if the slicing is performed in the increasing layer index order.
    """

    def __init__(self, *, do_slice_head: bool = False):
        super().__init__(do_slice_head=do_slice_head)

    @final
    def _get_attention_input_dimension(self, idx: int) -> int:
        # return the input embedding dimension when at the first attn layer inputs
        if idx == 0:
            return self.get_embedding_dimensions()[0]  # all dimensions are the same there

        return self.get_mlp_output_dimension(idx - 1)

    @final
    def _get_mlp_input_dimension(self, idx: int) -> int:
        return self.get_attention_output_dimension(idx)


class FunctionSlicingScheduler(ForwardSlicingScheduler):
    """
    A forward slicing scheduler that applies sparsity based on the provided function.
    """

    def __init__(
        self,
        *,
        mlp_sparsity_func: Callable[[float], float],
        attn_sparsity_func: Callable[[float], float] = None,
        round_interval: int = 1,
        do_slice_head: bool = False,
    ):
        super().__init__(do_slice_head=do_slice_head)
        self.mlp_sparsity: Callable[[float], float] = mlp_sparsity_func
        self.attn_sparsity: Callable[[float], float] = attn_sparsity_func  # unused for parallel blocks case
        self.round_interval: int = round_interval

    def _get_layer_dimension(self, idx: int, is_attn_layer: bool = False) -> int:
        loc = idx / (self.layers_num - 1)
        assert 0 <= loc <= 1
        sparsity = self.attn_sparsity(loc) if is_attn_layer else self.mlp_sparsity(loc)
        assert 0 <= sparsity < 1
        val = int(self.hidden_size * (1 - sparsity))
        val -= val % self.round_interval
        return val

    def _get_embedding_dimensions(self) -> dict[int, int]:
        return defaultdict(lambda: self._get_layer_dimension(0))

    def _get_attention_output_dimension(self, idx: int) -> int:
        return self._get_layer_dimension(idx, is_attn_layer=True)

    def _get_mlp_output_dimension(self, idx: int) -> int:
        return self._get_layer_dimension(idx + 1)  # head dimension matching ensures location will not exceed 1

    def _get_head_dimension(self) -> int:
        return self._get_layer_dimension(self.layers_num - 1)

    @staticmethod
    def create_linear(
        mlp_start: float,
        mlp_end: float,
        attn_start: float | None = None,
        attn_end: float | None = None,
        round_interval: int = 1,
        do_slice_head: bool = False,
    ) -> 'FunctionSlicingScheduler':
        """Create a linear slicing scheduler, mainly as an example for testing."""

        def linear(start: float, end: float) -> Callable[[float], float]:
            def linear_sparsity_func(location: float) -> float:
                return start + (end - start) * location

            return linear_sparsity_func

        return FunctionSlicingScheduler(
            mlp_sparsity_func=linear(mlp_start, mlp_end),
            attn_sparsity_func=linear(attn_start, attn_end)
            if (attn_start is not None and attn_end is not None)
            else None,
            round_interval=round_interval,
            do_slice_head=do_slice_head,
        )


class ExplainedVarianceSlicingScheduler(ForwardSlicingScheduler):
    """A slicing scheduler that applies sparsity based on the explained variance from the PCA."""

    def __init__(
        self,
        *,
        uev_threshold: float,
        round_interval: int = 1,
        do_slice_head: bool = False,
    ):
        super().__init__(do_slice_head=do_slice_head)
        self.uev_threshold: float = uev_threshold
        self.round_interval: int = round_interval

    def _get_layer_dimension(self, eigen_vals: list[float], plot: bool = False) -> int:
        eigen_vals = np.array(eigen_vals)
        cum_var = np.cumsum(np.array(eigen_vals)) / np.sum(eigen_vals)
        dim = np.argmax(cum_var > 1 - self.uev_threshold)
        dim -= dim % self.round_interval
        dim = int(dim)
        return dim

    def _get_embedding_dimensions(self) -> dict[int, int]:
        return defaultdict(lambda: self._get_layer_dimension(self.embedding_eigenvalues))

    def _get_attention_output_dimension(self, idx: int) -> int:
        return self._get_layer_dimension(self.mlp_eigenvalues[idx])

    def _get_mlp_output_dimension(self, idx: int) -> int:
        return self._get_layer_dimension(self.attention_eigenvalues[idx])

    def _get_head_dimension(self) -> int:
        return self.get_attention_output_dimension(self.layers_num - 1)
