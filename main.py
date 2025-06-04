import random
from loader import Loader
from os.path import join
import math
import random
import json

# loading up data
input_path = "C:/Users/blahb/.cache/kagglehub/datasets/hojjatk/mnist-dataset/versions/1"
training_images_path = join(input_path, "train-images-idx3-ubyte/train-images-idx3-ubyte")
training_labels_path = join(input_path, "train-labels-idx1-ubyte/train-labels-idx1-ubyte")
testing_images_path = join(input_path, "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
testing_labels_path = join(input_path, "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")

loader = Loader(training_images_path, training_labels_path, testing_images_path, testing_labels_path)
train_x, train_y = loader.packing_pairs(loader.training_images_path, loader.training_labels_path)
test_x, test_y = loader.packing_pairs(loader.test_images_path, loader.test_labels_path)

# start training
LAYERS = [784, 392, 196, 98, 49, 25, 10]
N = len(LAYERS)
w = [] # weights
dw = [] # dC / dw
dw_sum = []
b = [[random.random() * 10 - 5 for _ in range(LAYERS[i])] for i in range(len(LAYERS))] # biases
db = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))] # dC / db
db_sum = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))]
act = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))] # activations
da = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))] # dC / dact
da_sum = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))]
C = 0

# initializes w as randoms
for i in range(len(LAYERS)-1):
    m = [[random.random() * 2 - 1 for _ in range(LAYERS[i])] for _ in range(LAYERS[i+1])]
    dm = [[0 for _ in range(LAYERS[i])] for _ in range(LAYERS[i+1])]
    dm_sum = [[0 for _ in range(LAYERS[i])] for _ in range(LAYERS[i+1])]
    w.append(m)
    dw.append(m)
    dw_sum.append(dm_sum)

def sig(x): 
    return 1 / (1 + math.exp(-x))

def sig_p(x):
    return math.exp(-x) / pow(1 + math.exp(-x), 2)

def dot(x, y): # returns dot product of two vectors
    return sum(x[i]*y[i] for i in range(len(x)))
    
def clear_activations(): # resets activations
    for i in range(len(LAYERS)):
        for j in range(LAYERS[i]):
            act[i][j] = 0
    
def clear_differentials(): # resets differentials of w, b, and a
    for i in range(len(LAYERS)-1):
        for j in range(LAYERS[i+1]):
            for k in range(LAYERS[i]):
                dw[i][j][k] = 0
    for i in range(len(LAYERS)):
        for j in range(LAYERS[i]):
            da[i][j] = db[i][j] = 0

def clear_sums():
    for i in range(len(LAYERS)-1):
        for j in range(LAYERS[i+1]):
            for k in range(LAYERS[i]):
                dw_sum[i][j][k] = 0
    for i in range(len(LAYERS)):
        for j in range(LAYERS[i]):
            da_sum[i][j] = db_sum[i][j] = 0

def forward_propagation(image, label, id):
    clear_activations() # reset activations
    clear_differentials() # resets differentials
    assert(len(image) == 784)
    for i in range(LAYERS[0]):
        act[0][i] = image[i] / 255.0 # convert grayscale to float
    for i in range(len(LAYERS)-1):
        for j in range(LAYERS[i+1]): # compute next neuron's activation
            act[i+1][j] = sig(dot(act[i], w[i][j]) + b[i+1][j])
    C = 0 
    for i in range(LAYERS[N-1]):
        C += pow(act[N-1][i] - (1 if label == i else 0), 2)
    bestGuess = 0
    for i in range(10):
        if act[N-1][i] > act[N-1][bestGuess]:
            bestGuess = i
    print(f"Cost for batch {id} = {C}, best guess is {bestGuess}")

BATCH_SIZE = 20

def back_propagation(image, label, id):
    # start with last layer's dC / da
    for i in range(LAYERS[N-1]):
        da[N-1][i] = 2 * pow(act[N-1][i] - (1 if label==i else 0), 2)
    for i in range(N-2, -1, -1):
        for j in range(LAYERS[i+1]): # differentials for b
            db[i+1][j] = da[i+1][j]
        for j in range(LAYERS[i+1]): # differentials for w and a
            for k in range(LAYERS[i]):
                dw[i][j][k] = da[i+1][j] * sig_p(act[i][k] * w[i][j][k] + b[i+1][j]) * act[i][k]
                da[i][k] += da[i+1][j] * sig_p(act[i][k] * w[i][j][k] + b[i+1][j]) * w[i][j][k]


def contribute_to_sum():
    for i in range(len(LAYERS)):
        for j in range(LAYERS[i]):
            da_sum[i][j] += (1.0 / BATCH_SIZE) * da[i][j]
            db_sum[i][j] += (1.0 / BATCH_SIZE) * db[i][j]
    for i in range(N-2):
        for j in range(LAYERS[i+1]):
            for k in range(LAYERS[i]):
                dw_sum[i][j][k] += (1.0 / BATCH_SIZE) * dw[i][j][k]

def gradient_descent(step_size):
    for i in range(N-1):
        for j in range(LAYERS[i+1]):
            for k in range(LAYERS[i]):
                w[i][j][k] -= step_size * dw_sum[i][j][k]
    for i in range(N):
        for j in range(LAYERS[i]):
            b[i][j] -= step_size * db_sum[i][j]
    
STEP_SIZE = 0.1 # try first with constant step sizes

for id, data in enumerate(zip(train_x, train_y)):
    image, label = data

    # every BATCH_SIZE, perform a mini-batch gradient descent
    if id % BATCH_SIZE == 0:
        gradient_descent(STEP_SIZE)
        clear_sums()
    
    forward_propagation(image, label, id)

    # every 2*BATCH_SIZE, print an image and see if the prediction is right
    if id % (2*BATCH_SIZE) == 0:
        for i in range(784):
            print("#" if image[i] > 125 else '.', end = "\n" if i % 28 == 27 else "")
        print(f"correct answer is {label}")

    # every 100, save the current weights and biases into a txt file
    if id % 100 == 0:
        with open("weights.txt", "w") as file:
            json.dump(w, file, indent=4)
        with open("biases.txt", "w") as file:
            json.dump(b, file, indent=4)
        
    back_propagation(image, label, id)
    contribute_to_sum()