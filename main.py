import argparse

import numpy as np
from matplotlib import pyplot as plt

from Errors import MSE
from MNistClassifier import MNistClassifier
from dataset import Dataset

# Variables
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='mnist')
parser.add_argument('--lr', type=float, default=.001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)
args = parser.parse_args()


# init the model
model = MNistClassifier()
model.randomize()
loss = MSE()
data = Dataset(args.dataset_path, args.batch_size)


losses = []
# Train the model
for epoch in range(args.epochs):
    for iteration, (image, label) in enumerate(data):
        # forward pass
        y = model.forward(image)
        loss_value = loss.forward(y, label)

        # backward pass
        gradient_y = loss.backward()
        gradient_y = gradient_y
        model.backward(gradient_y)
        model.optimize(args.lr)

        print(f"Epoch: {epoch},"
              f"Iteration: {iteration},"
              f"Loss: {loss_value:.6f}"
              )
        losses.append(loss_value)

# evaluate the model
data_test = Dataset(args.dataset_path, args.batch_size, True)
pred_test = model.forward(data_test[:][0])
loss_value = loss.forward(pred_test, data_test[:][1])
loss_value = loss_value.mean()
print(f"Evaluation Loss: {loss_value:.6f}")

# Plot example predictions
plt.figure(dpi=200)
plt.rcParams.update({'font.size': 8})  # Set the font size to a smaller value
for i in range(18):
    plt.subplot(3, 6, i + 1)
    plt.axis('off')
    plt.imshow(data_test[i][0].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {np.argmax(pred_test[i])} ( {np.argmax(data_test[i][1])} )")
plt.show()


# Plot the loss
plt.plot(np.log(losses))
plt.show()