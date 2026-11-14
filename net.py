import cupy as np
import time
from dataset import make_circles, make_moons, make_spirals
class Layer:
    def __init__(self,weights, biases, activation, derivative_activation, momentum=0.9, variance=0.999,eps=1e-8, weight_decay = 0.00001 ):
        self.weights= np.array(weights)
        self.biases = np.array(biases)
        self.activation = activation
        self.derivative = derivative_activation

        self.weight_decay = weight_decay
        self.momentum = momentum
        self.step = 0
        self.variance = variance
        self.eps = eps

        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.biases)

        self.weight_variance = np.zeros_like(self.weights)
        self.bias_variance = np.zeros_like(self.biases)



    def forward(self, inputs):
        self.input = inputs
        self.weighted_sum = np.dot(inputs, self.weights) + self.biases

        return self.activation(self.weighted_sum)
    
    def train(self, incoming_blame, learning_rate):
        batch_size = incoming_blame.shape[0]
        
        activation_derivative = self.derivative(self.weighted_sum)
        delta = incoming_blame * activation_derivative

        grad_weight = np.dot(self.input.T, delta) / batch_size
        grad_bias = np.sum(delta, axis=0) / batch_size

        blame = np.dot(delta, self.weights.T)

        #Gradient clipping: Stoppar gradient från o bli piss stor
        grad_norm_w = np.linalg.norm(grad_weight)
        if grad_norm_w > 1.0:
            grad_weight = grad_weight * (1.0 / grad_norm_w )

        grad_norm_b = np.linalg.norm(grad_bias)
        if grad_norm_b > 1.0:
            grad_bias = grad_bias*(1.0/grad_norm_b)



        # Adam och momentum

        #Adam: jämnar ut  vikter / biases med mycket lutning

        #momentum: kollar historiken om den fortfarande rör sig åt samma håll puttar den lite mer
        self.step +=1

        
        self.weight_momentum = self.momentum * self.weight_momentum + (1.0-self.momentum) *grad_weight
        self.bias_momentum = self.momentum * self.bias_momentum + (1.0 - self.momentum) *grad_bias

        self.weight_variance = self.variance *self.weight_variance +(1.0-self.variance)* (grad_weight**2)
        self.bias_variance = self.variance *self.bias_variance + (1.0-self.variance) * (grad_bias**2)

        momentum_hat_weight = self.weight_momentum / (1.0 - self.momentum**self.step)
        momentum_hat_bias = self.bias_momentum / (1.0 - self.momentum**self.step)

        variance_hat_weight = self.weight_variance / (1.0 - self.variance **self.step)
        variance_hat_bias = self.bias_variance / (1.0 - self.variance **self.step)

        #AdamW: Ger weight decay så inget fastnar i sina vägar
        self.weights -= learning_rate * (momentum_hat_weight/ (np.sqrt(variance_hat_weight) +self.eps) +self.weight_decay *self.weights)
        self.biases -= learning_rate * (momentum_hat_bias / (np.sqrt(variance_hat_bias) + self.eps) +self.weight_decay *self.biases )
            
        return blame
        



class Network:
    def __init__(self, layers):
        self.layers = layers


    def calculate(self, input):
        layers = self.layers
        output = input
        for layer in layers:
            output = layer.forward(output)
        return output


    def backward(self, output, expected, learning_rate):
        blame = output - expected
        for layer in reversed(self.layers):
            blame = layer.train(blame, learning_rate)



def sigmoid_derivative(x):
    s = Sigmoid(x)
    return s * (1.0 - s)

def LeakyReLU(x):
    return np.where(x > 0, x, 0.01 * x)

def LeakyReLU_derivative(x):
    return np.where(x > 0, 1.0, 0.01)

def Sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

def linear_derivative(x):
    return 1.0

np.random.seed(43)

def rand_network(input, layers, output):
    network_layers = []
    past = input

    for layer in layers:
        # In Layer creation:
        weights = (np.random.randn(past, layer) * np.sqrt(2 / past)).astype(np.float32)
        biases = np.zeros(layer, dtype=np.float32) 

        net = Layer(weights, biases, LeakyReLU,  LeakyReLU_derivative)
        network_layers.append(net)

        past = layer

    weights = (np.random.randn(past, output) * np.sqrt(1 / past)).astype(np.float32)
    biases = np.zeros(output, dtype=np.float32)

    last = Layer(weights, biases, Sigmoid, linear_derivative)
    network_layers.append(last)

    network = Network(network_layers)

    return network

def distance_check(network, examples, correct):
    
    prediction = network.calculate(examples)
    
    return np.mean(np.abs(prediction-correct))




examples = np.array([
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 1]
], dtype=np.float32)

correct = np.array([
    [1],
    [0],
    [0],
    [1]
], dtype=np.float32)


examples = np.asarray(examples)
correct = np.asarray(correct)



epoch = 0
rounds = 1000
initial_learningrate = 0.05
min_learningrate = 0.001
warmup_rounds = 10



network = rand_network(2, [10, 10], 1)

start_time = time.perf_counter()

while epoch < rounds:
    #warmup rounnds to build variety and momentum
    if epoch < warmup_rounds:
        current_lr = initial_learningrate * (epoch + 1) / warmup_rounds
    else:
            #Cosine Decay phase Slowly slow down learningrate to make it over time finetune more
        current_lr = min_learningrate + 0.5 * (initial_learningrate - min_learningrate) * (1.0 + np.cos(np.pi * epoch / rounds))

    
    
    output = network.calculate(examples)
    network.backward(output, correct, current_lr)
    if distance_check(network, examples, correct) < 0.01:
        print(f"got done EARLY!! finished at epoch: #{epoch}")
        break
    epoch += 1
    if epoch % 1000 == 0:
        print(f"trained #{epoch} amounts of times inacuracy is {distance_check(network, examples, correct)}")
        
    


   # print(network.calculate(examples))

end_time = time.perf_counter()

duration = end_time - start_time
print(f"Training took: {duration:.2f} seconds and inacuracy is {distance_check(network, examples, correct)} finished at epoch #{epoch}")