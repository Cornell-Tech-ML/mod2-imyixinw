"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

from audioop import bias
from numpy import stack
import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5.
# Implement a neural network over the data with three linears
# (2-> Hidden (relu), Hidden -> Hidden (relu), Hidden -> Output (sigmoid)).
# It should do exactly the same thing as the corresponding functions
# in project/run_scalar.py, but now use the tensor code base.



class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        middle = self.layer1.forward(x).relu()
        end = self.layer2.forward(middle).relu()
        return self.layer3.forward(end).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)

    def forward(self, inputs):
        # output = inputs * weight + bias
        batch_size, in_size = inputs.shape
        output = minitorch.zeros((batch_size, self.out_size))

        for i in range(batch_size):
            for j in range(self.out_size):
                dot_product = 0.0
                for k in range(self.in_size):
                    dot_product += inputs[i, k] * self.weights.value[k, j]

                output[i, j] = dot_product

        for i in range(batch_size):
            for j in range(self.out_size):
                output[i, j] += self.bias.value[j]

        return output



def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
