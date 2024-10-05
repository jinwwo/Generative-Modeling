import matplotlib.pyplot as plt
import torch

from model import Discriminator, Generator


def generate(device="cuda:0"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        G = Generator()
        G.load_state_dict(torch.load("./Generator.pth", map_location=device))
        feature_vector = torch.randn(1, 100, 1, 1).to(device)
        pred = G(feature_vector).squeeze()
        pred = pred.permute(1, 2, 0).cpu().numpy()

        plt.imshow(pred)
        plt.title("predicted image")
        plt.show()


if __name__ == "__main__":
    generate()