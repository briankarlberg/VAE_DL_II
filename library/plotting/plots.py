import matplotlib.pyplot as plt
from pathlib import Path

def plot_model_performance(history: object, sub_directory: str, file_name: str):
    plt.figure(num=None, figsize=(6, 4), dpi=90)
    for key in history.history:
        plt.plot(history.history[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    save_path = Path(f"{file_name}.png")
    plt.savefig(save_path)
    plt.close()