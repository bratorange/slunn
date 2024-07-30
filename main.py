import argparse

import numpy as np
from matplotlib import pyplot as plt

from Errors import MSE, CrossEntropyLoss
from MNistClassifier import MNistClassifier
from dataset import Dataset

# Variables
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='MNIST')
parser.add_argument('--initial_lr', type=float, default=.02)
parser.add_argument('--lr_decay', type=float, default=.05)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
args = parser.parse_args()

# init the model
model = MNistClassifier()
model.randomize()
loss = CrossEntropyLoss()
data_train = Dataset(args.dataset_path, args.batch_size)
data_test = Dataset(args.dataset_path, args.batch_size, True)


def evaluate():
    global pred_test, error
    pred_test = model.forward(data_test[:][0])
    error = 1 - np.mean(pred_test.argmax(axis=1) == data_test[:][1].argmax(axis=1))
    return error


losses = []
error_rate = []
t = 0
# Train the model
for epoch in range(args.epochs):
    lr = args.initial_lr * np.exp(-args.lr_decay * epoch)
    for iteration, (image, label) in enumerate(data_train):
        # forward pass
        y = model.forward(image)
        error = loss.forward(y, label)

        # backward pass
        gradient_y = loss.backward()
        gradient_y = gradient_y
        model.backward(gradient_y)
        model.optimize(lr)

        # if iteration % 10000 == 0:
        #     print(f"Epoch: {epoch}, "
        #           f"Iteration: {iteration}, "
        #           f"Loss: {error:.6f}, "
        #           f"Learning Rate: {lr:.6f}, "
        #           )
        losses.append((t, error,))
        t += 1
    error_rate.append((t, evaluate(),))
    print(f"Finished epoch: {epoch}, Evaluation Loss: {error_rate[-1][1]:.6f}, Learning Rate: {lr:.6f}")

# evaluate the model
pred_test = model.forward(data_test[:][0])
print(f"Evaluation Error: {evaluate():.6f}")

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
losses = np.array(losses).transpose()
error_rate = np.array(error_rate).transpose()
# plt.plot(losses[0], np.log(losses[1]), '-', label='Training Loss', markersize=.5)
plt.plot(error_rate[0], np.log(error_rate[1]), 'o-', label='Evaluation Error', markersize=1)
plt.legend()
plt.show()
