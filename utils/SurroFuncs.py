import torch
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based.surrogate import SurrogateFunctionBase
from spikingjelly.activation_based.auto_cuda import cfunction

class boxcar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, width):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.width = width
        return surrogate.heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        return boxcar_backward(grad_output, ctx.saved_tensors[0], ctx.width)

class Boxcar(SurrogateFunctionBase):
    def __init__(self, width=1.0, spiking=True):
        """
        :param width: Width of the boxcar function
        :param spiking: Whether to output spikes. The default is ``True``, using ``heaviside`` in forward propagation and using surrogate gradients in back propagation. If ``False``,
            surrogate gradients are not used. During forward propagation, the gradient during back propagation is used to replace the original function

        The boxcar surrogate spiking function. The gradient is defined by

        .. math::
            g'(x) = \\begin{cases}
                \\frac{1}{width}, & -\\frac{width}{2} \\leq x \\leq \\frac{width}{2} \\\\
                0, & \\text{otherwise}
            \\end{cases}

        The primitive function is defined by

        .. math::
            g(x) = \\begin{cases}
                0, & x < -\\frac{width}{2} \\\\
                \\frac{x}{width} + \\frac{1}{2}, & -\\frac{width}{2} \\leq x \\leq \\frac{width}{2} \\\\
                1, & x > \\frac{width}{2}
            \\end{cases}
        """
        super().__init__(width, spiking)

    @staticmethod
    def spiking_function(x, width):
        return boxcar.apply(x, width)

    @staticmethod
    @torch.jit.script
    def primitive_function(x: torch.Tensor, width: float):
        return torch.clamp(x / width + 0.5, 0.0, 1.0)

    @staticmethod
    def backward(grad_output, x, width):
        return boxcar_backward(grad_output, x, width)[0]

    def cuda_code(self, x: str, y: str, dtype='fp32'):
        sg_name = 'sg_' + self._get_name()
        width = str(self.alpha) + 'f'
        code = f'''
        {self.cuda_code_start_comments()}
        '''

        if dtype == 'fp32':
            code += f'''
        const float {sg_name}_abs_x = fabsf({x});
        const float {y} = {sg_name}_abs_x <= {width}/2.0f ? 1.0f/{width} : 0.0f;
        '''
        elif dtype == 'fp16':
            code += f'''
        const half2 {sg_name}_width = __float2half2_rn({width});
        const half2 {sg_name}_abs_x = __habs2({x});
        const half2 {y} = __hle2({sg_name}_abs_x, __h2div({sg_name}_width, __float2half2_rn(2.0f))) ? __h2div(__float2half2_rn(1.0f), {sg_name}_width) : __float2half2_rn(0.0f);
        '''
        else:
            raise NotImplementedError

        code += f'''
        {self.cuda_code_end_comments()}
        '''
        return code

    def cuda_codes(self, y: str, x: str, dtype: str):
        return cfunction.boxcar_backward(y=y, x=x, width=self.alpha, dtype=dtype)

@torch.jit.script
def boxcar_backward(grad_output: torch.Tensor, x: torch.Tensor, width: float):
    mask = (x.abs() <= width / 2)
    grad_x = torch.where(mask, torch.ones_like(x) / width, torch.zeros_like(x))
    return grad_output * grad_x, None