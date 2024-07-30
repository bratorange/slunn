import argparse

from matplotlib import pyplot as plt

from Errors import MSE
from MNistClassifier import MNistClassifier
from dataset import Dataset

# Variables
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='mnist')
parser.add_argument('--lr', type=float, default=.001)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()


# init the model
model = MNistClassifier()
model.randomize()
loss = MSE()
data = Dataset(args.dataset_path, args.batch_size)

# create a linear function as training data
# images = np.linspace(0, 3, 1000).reshape(-1, 1)
# labels = np.sin(images)
# labels = labels + np.random.normal(0, .01, labels.shape)



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

print(model.get_parameters())

plt.figure(dpi=200)
plt.plot(*data[:], 'ro', markersize=1)
plt.plot(data[:][0], model.forward(data[:][0]), 'b', linewidth=1)
plt.show()

# Plot the loss
plt.plot(losses)
plt.show()