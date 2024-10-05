import torch
import torch.nn as nn
import tqdm
from PIL import Image
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder

from GAN.utils import _TRANSFORMS, parse_arguments, set_cuda_device
from model import Discriminator, Generator


def load_data(data_path, batch_size=128):
    transforms = _TRANSFORMS

    dataset = ImageFolder(
        root=data_path,
        transform=transforms,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train():
    args = parse_arguments()
    set_cuda_device(args.device)

    device_ids = [int(d) for d in args.device.split(",")]
    device = torch.device(
        f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"
    )

    G = Generator()
    D = Discriminator()
    G.apply(weights_init)
    D.apply(weights_init)

    if len(device_ids) > 1:
        G = nn.DataParallel(G, device_ids=device_ids)
        D = nn.DataParallel(D, device_ids=device_ids)

    data_path = args.data_path
    batch_size = args.batch_size
    num_epochs = args.epoch
    dataloader = load_data(data_path, batch_size)

    train_loop(D, G, num_epochs, dataloader, device)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_loop(D, G, num_epochs, dataloader, device):
    G_optim = Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    D_optim = Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        iterator = tqdm.tqdm(enumerate(dataloader, 0), total=len(dataloader))

        for i, data in iterator:
            D_optim.zero_grad()

            label = torch.ones_like(data[1], dtype=torch.float32).to(device)
            label_fake = torch.zeros_like(data[1], dtype=torch.float32).to(device)
            
            real = D(data[0].to(device))
            Dloss_real = nn.BCELoss()(torch.squeeze(real), label)
            Dloss_real.backward()

            noise = torch.randn(label.shape[0], 100, 1, 1, device=device)
            
            fake = G(noise)
            output = D(fake.detach())

            Dloss_fake = nn.BCELoss()(torch.squeeze(output), label_fake)
            Dloss_fake.backward()
            Dloss = Dloss_real + Dloss_fake
            D_optim.step()

            G_optim.zero_grad()
            output = D(fake)
            
            Gloss = nn.BCELoss()(torch.squeeze(output), label)
            Gloss.backward()
            G_optim.step()

            iterator.set_description(
                f"epoch:{epoch} iteration:{i} D_loss:{Dloss} G_loss:{Gloss}"
            )


if __name__ == "__main__":
    train()
