import torch
from torch.utils.data import DataLoader
from src.data.dataset import FERDataset
from src.models.emotion_model import EmotionModel

dataset = FERDataset("data/raw/fer2013.csv")

loader = DataLoader(dataset,batch_size=32,shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionModel().to(device)

optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)

loss_fn = torch.nn.CrossEntropyLoss()

epochs = 10

for epoch in range(epochs):

    for images,labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = loss_fn(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epoch:",epoch,"Loss:",loss.item())

torch.save(model.state_dict(),"models/emotion_vit.pth")
