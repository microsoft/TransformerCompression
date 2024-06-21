import torch

from ..quant_utils import PackedQuantizedTensor, dequantize, to_int4, from_int4


class QuarotFP16Linear(torch.nn.Module):
    """
    Linear module for emulating quantized weights and activations. All tensors are stored in FP16.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        bias: bool = False,
        offset: bool = False,
        group_size: int = None,
        dtype=torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert bits in [2, 3, 4, 8, 16], "Unsupported number of bits"
        self.bits = bits
        self.dtype = dtype

        # figure out group size
        if group_size is None:
            group_size = in_features  # tensor-level scaling, one effective group
        assert in_features % group_size == 0, "implied number of groups must be an integer!"
        self.group_size = group_size

        # register necessary buffers
        self.register_buffer('weight_scales', torch.ones((self.out_features, in_features // group_size), dtype=dtype))
        self.register_buffer('weight', torch.zeros((self.out_features, self.in_features), dtype=dtype))
        if bias:
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
        if offset:
            self.register_buffer('offset', torch.zeros((self.out_features, in_features // group_size), dtype=dtype))
        else:
            self.offset = None
            
    def compress(self):
        """
        Store the weight (and offsets) of the layer in lower memory form, 
        according to the settings in config. This method assumes that some quantization scheme has been applied,
        so the weights are already integers, but stored FP16.
        """
        assert self.weight.dtype == self.dtype, "Weights must be in FP to compress"
        is_signed = self.offset is None
        if self.bits == 16:
            return # nothing to do
        elif self.bits == 8:
            dtype = torch.int8 if is_signed else torch.uint8
            self.weight = self.weight.int().to(dtype)
            if self.offset is not None:
                self.offset = self.offset.int().to(dtype)
        elif self.bits in [2, 3, 4]:
            # we pack in bit-width 4 here. This is a bit wasteful for 2 and 3 bit quantization, but it's simpler.
            self.weight = to_int4(self.weight, signed = is_signed)
            if self.offset is not None:
                if self.offset.shape[-1] == 1:
                    pass # do not pack the offset if there's only a single group: we need two+ columns to pack/unpack
                else:
                    self.offset = to_int4(self.offset, signed = is_signed)
                
    def decompress(self):
        """
        Restore the weight (and offsets) of the layer to FP16 form.
        """
        is_signed = self.offset is None # offset imples asymm. Asmm is not signed, symm in signed
        if self.weight.dtype == self.dtype:
            # wither we're just doing 16 bits, or we've not quantized yet.
            weight, offset = self.weight, self.offset
        elif self.bits == 8:
            weight = self.weight.to(self.dtype)
            offset = self.offset.to(self.dtype) if self.offset is not None else None
        elif self.bits in [2, 3, 4]:
            weight = from_int4(self.weight, signed=is_signed).to(self.dtype)
            if self.offset is not None:
                if self.offset.dtype == self.dtype:
                    offset = self.offset # we didn't pack offset because it had only one column
                else:
                    offset = from_int4(self.offset, signed=is_signed).to(self.dtype)
            else:
                offset = None
        return weight, self.weight_scales, offset

    def forward(self, x: PackedQuantizedTensor) -> torch.tensor:
        # de-quantize the activations
        x = dequantize(x.quantized_x, x.scales_x, offset=x.offset)

        # de-quantize the weights
        weight, scales, offset = self.decompress()
        W = dequantize(weight, scales, offset)

        #  run standard linear on dequantized weights and activations
        return torch.functional.F.linear(x, W, self.bias)

    @classmethod
    def like(cls: type, module: torch.nn.Linear, bits: int, groupsize: int = None, offset: bool = False) -> 'QuarotFP16Linear':
        '''
        Generate a new QuarotFP16Linear module with the same shapes & bias flag as a given Linear module.
        '''
        return cls(
            module.in_features, module.out_features, bits=bits, bias=module.bias is not None, group_size=groupsize, offset=offset
        )
        
