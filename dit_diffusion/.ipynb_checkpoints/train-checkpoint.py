import dit_diffusion.config as config
from torch.utils.data import DataLoader
import torch 
from torch import nn 
from .dataset import MNIST
from .diffusion import forward_add_noise
from .dit import DiT
import os
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

def main():
    dataset = MNIST()
    model = DiT(img_size=28, patch_size=4, channel=1, emb_size=64, label_num=10, dit_num=3, head=4).to(config.DEVICE)

    try:
        model.load_state_dict(torch.load('model.pth'))

    except:

        pass 

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=10, persistent_workers=True)

    model.train()

    iter_count = 0

    for epoch in range(config.EPOCH):
        for imgs,labels in dataloader:
            x = imgs * 2 - 1
            t = torch.randint(0, config.T, (imgs.size(0), ))
            y = labels

            x, noise = forward_add_noise(x, t)
            pred_noise = model(x.to(config.DEVICE), t.to(config.DEVICE), y.to(config.DEVICE))

            loss = loss_fn(pred_noise, noise.to(config.DEVICE))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_count % 1000 == 0:
                print(f'epoch:{epoch} iter:{iter_count}, loss:{loss}')

                torch.save(model.state_dict(), '.model.pth')

                os.replace('.model.pth', 'model.pth')

            iter_count += 1
            
def train_model(model, dataloader, epoch_num, loss_fn, optimizer):
    model.train()
    losses = []

    iter_count = 0
    total_loss = 0
    batch_count = 0

    for epoch in tqdm(range(epoch_num)):
        for imgs, labels in dataloader:
            x = imgs * 2 - 1
            t = torch.randint(0, config.T, (imgs.size(0), ))
            y = labels

            x, noise = forward_add_noise(x, t)
            pred_noise = model(x.to(config.DEVICE), t.to(config.DEVICE), y.to(config.DEVICE))

            loss = loss_fn(pred_noise, noise.to(config.DEVICE))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1 
            
            if iter_count % 10000 == 0:
                print(f'epoch:{epoch} iter:{iter_count}, loss:{loss}')

                torch.save(model.state_dict(), '.model.pth')
                os.replace('.model.pth', 'model.pth')

            iter_count += 1

        avg_loss = total_loss / batch_count 
        losses.append(avg_loss) 
        total_loss = 0
        batch_count = 0 

    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(range(epoch_num), losses, marker='o')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()