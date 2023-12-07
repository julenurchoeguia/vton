#%%
from datasets import load_dataset
from lib_refiner_autoencoder import *
import torch
import matplotlib as plt

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("cats_vs_dogs", split="train")
dataset_train = dataset[:int(len(dataset)*0.8)]
dataset_test = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
dataset_validation = dataset[int(len(dataset)*0.9):]
#%%
first_example = dataset_train[0]
image_data = first_example['image'] 
plt.figure()
plt.imshow(image_data)
plt.show()
# print(len(dataset_train))
# print(len(dataset_test))
# print(len(dataset_validation))

#%%
"""
autoencoder = AutoEncoder()
load_dropout(autoencoder, dropout=0.5)
autoencoder.to(device)
autoencoder.train()
lr = 1e-4
num_epochs = 100

optimizer = torch.optim.Adam(autoencoder.parameters() , lr=lr)
for epoch in range(num_epochs):
    loss_iter = 0
    for image in dataset_train:
        image = utils.image_to_tensor(image).to(device)
        y = autoencoder(image)
        loss = (y-image).norm()
        loss_iter += loss
    loss = loss_iter/len(dataset_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"epoch {epoch} : loss {loss.item()}")

autoencoder.eval()
for image in dataset_test:
    image = utils.image_to_tensor(image).to(device)
    result = autoencoder(image)
    utils.tensor_to_image(image.data).show()
    utils.tensor_to_image(result.data).show()

"""