import torch

def estimate_gradient_size_MB(model, input_shape, device="cpu"):
    """
    Ước lượng kích thước gradient truyền về (tức kích thước output cuối của model).

    Args:
        model: nn.Module (client, edge, or cloud model)
        input_shape: tuple, ví dụ (3, 32, 32)
        device: 'cuda' hoặc 'cpu'

    Returns:
        size_MB: float - kích thước output cuối cùng theo MB
    """
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_shape).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    numel = output.numel()
    element_size = output.element_size()  # thường là 4 bytes (float32)
    size_MB = (numel * element_size) / (1024**2)
    return size_MB