import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


# Make results reproducible
torch.manual_seed(42)


# --------------------------------------------------
# 1. Load the Cora dataset
# --------------------------------------------------

dataset = Planetoid(
    root="data/Cora",
    name="Cora"
)

data = dataset[0]


print("Dataset:", dataset)
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Number of features:", dataset.num_features)
print("Number of classes:", dataset.num_classes)


# --------------------------------------------------
# 2. Create the GNN
# --------------------------------------------------

class GNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.gcn1 = GCNConv(
            dataset.num_features,
            16
        )

        self.gcn2 = GCNConv(
            16,
            dataset.num_classes
        )

    def forward(self, x, edge_index):

        x = self.gcn1(x, edge_index)

        x = F.relu(x)

        x = self.gcn2(x, edge_index)

        return x


# --------------------------------------------------
# 3. Create model
# --------------------------------------------------

model = GNN()


# --------------------------------------------------
# 4. Create optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4
)


# --------------------------------------------------
# 5. Train
# --------------------------------------------------

for epoch in range(200):

    model.train()

    optimizer.zero_grad()

    output = model(
        data.x,
        data.edge_index
    )

    loss = F.cross_entropy(
        output[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()

    optimizer.step()

    if epoch % 20 == 0:

        predictions = output.argmax(dim=1)

        correct = (
            predictions[data.train_mask]
            == data.y[data.train_mask]
        )

        train_accuracy = correct.float().mean()

        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Train Accuracy: {train_accuracy.item():.4f}"
        )


# --------------------------------------------------
# 6. Test
# --------------------------------------------------

model.eval()

with torch.no_grad():

    output = model(
        data.x,
        data.edge_index
    )

    predictions = output.argmax(dim=1)

    correct = (
        predictions[data.test_mask]
        == data.y[data.test_mask]
    )

    test_accuracy = correct.float().mean()


print()
print("Test Accuracy:", test_accuracy.item())


# --------------------------------------------------
# 7. Look at some individual predictions
# --------------------------------------------------

test_nodes = torch.where(data.test_mask)[0]

print()
print("Example predictions:")
print()

for node in test_nodes[:10]:

    predicted = predictions[node].item()
    actual = data.y[node].item()

    print(
        f"Node {node.item():4d} | "
        f"Predicted: {predicted} | "
        f"Actual: {actual}"
    )