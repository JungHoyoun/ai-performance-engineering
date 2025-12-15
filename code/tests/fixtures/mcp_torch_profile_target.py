import torch


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(512, 512, device=device)
    y = x @ x
    if device == "cuda":
        torch.cuda.synchronize()
    _ = y.sum().item()


if __name__ == "__main__":
    main()
