import numpy as np
import networkx as nx
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

class PRNNCell:
    def __init__(self, id, activation=None):
        self.id = id
        self.activation = activation if activation else lambda x: x
        self.input_value = 0
        self.output_value = 0
    
    def process(self, input_value):
        self.input_value = input_value
        self.output_value = self.activation(input_value)
        return self.output_value

class PRNN:
    def __init__(self, num_cells):
        self.graph = nx.DiGraph()
        self.cells = {i: PRNNCell(i, activation=np.tanh) for i in range(num_cells)}
        self._build_initial_structure()

    def _build_initial_structure(self):
        # Connect cells randomly for initial structure
        for i in range(len(self.cells)):
            for j in range(i + 1, len(self.cells)):
                if np.random.rand() < 0.5:  # 50% chance to connect
                    self.graph.add_edge(i, j, weight=np.random.rand())

    def forward(self, inputs):
        # Map inputs to the first few cells
        for i, value in enumerate(inputs):
            self.cells[i].process(value)
        
        # Propagate through the graph
        for u, v, data in self.graph.edges(data=True):
            weight = data['weight']
            input_value = self.cells[u].output_value * weight
            self.cells[v].process(self.cells[v].input_value + input_value)
        
        # Collect outputs from the last cells
        outputs = [cell.output_value for cell in self.cells.values()]
        return outputs

    def predict(self, inputs):
        outputs = self.forward(inputs)
        return np.argmax(outputs)

# Load and preprocess the MNIST dataset
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0
    return x_train, y_train, x_test, y_test

# Training PRNN
def train_prnn(prnn, x_train, y_train, epochs=10, learning_rate=0.01):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            # Forward pass
            outputs = prnn.forward(x)
            
            # Calculate loss and update (gradient-free for now)
            target = np.zeros(len(outputs))
            target[y] = 1
            loss = np.sum((outputs - target) ** 2)
            
            # Print progress
            if i % 1000 == 0:
                print(f"Step {i}: Loss = {loss:.4f}")

# Experiment
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    prnn = PRNN(num_cells=100)  # Small architecture with 100 cells
    train_prnn(prnn, x_train[:500], y_train[:500])  # Train on a small subset for a quick test
    
    # Evaluate on test set
    predictions = [prnn.predict(x) for x in x_test[:100]]
    accuracy = accuracy_score(y_test[:100], predictions)
    print(f"Test Accuracy: {accuracy:.2%}")
