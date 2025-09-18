from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First Convolution Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3), # input: 28x28x1, output: 26x26x8, RF = 3
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3), # input: 26x26x8, output: 24x24x16, RF = 5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # nn.Conv2d(16, 32, 3), # input: 24x24x16, output: 22x22x32, RF = 7
            # nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2), # output: 12x12x16, 
            nn.Dropout(0.25)
        )
        
        # Second Convolution Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 1), # input: 12x12x16, output: 12x12x32, RF  = 
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, 3), # input: 12x12x32, output: 10x10x16, RF = 18
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 8, 3), # input: 10x10x16, output: 8x8x8, RF = 18
            nn.ReLU(),
            nn.BatchNorm2d(8),            
            # nn.MaxPool2d(kernel_size=2, stride=2),# output: 5x5x16, RF = 28
            nn.Dropout(0.25)    
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 1), # input: 8x8x8, output: 8x8x16, RF = 28
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.25) ,   
            nn.Conv2d(16, 8, 3), # input: 8x8x16, output: 6x6x8, RF = 32
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.25)    
        )
        # Output Block
        self.fc = nn.Linear(8*6*6, 10) # 400 -> 10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 8*6*6)#16*9*9
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 16*5*5)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
        

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'epoch = {epoch} loss={loss.item()} batch_id={batch_idx}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(1)
    batch_size = 64

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                            transform=transforms.Compose([
                            transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
                            transforms.Resize((28, 28)),
                            transforms.RandomRotation((-15., 15.), fill=0),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Print model summary
    summary(model, input_size=(1, 28, 28))

    for epoch in range(1, 20):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()