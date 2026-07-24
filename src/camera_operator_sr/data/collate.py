import torch


def collate_frames(samples: list[dict]) -> dict:
    def stack(values):
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values)
        if isinstance(values[0], dict):
            return {key: stack([value[key] for value in values]) for key in values[0]}
        return values
    return stack(samples)
