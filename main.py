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
LAYERS = [784, 100, 10]
N = len(LAYERS)
w = [] # weights
dw = [] # dC / dw
dw_sum = []
b = [] # biases; start at 0
db = [] # dC / db
db_sum = []
act = [] # activations
da = [] # dC / dact
da_sum =[]
C = 0
STARTING_BATCH = 0

def load_initialization():
    global w, dw, dw_sum, b, db, db_sum, act, da, da_sum, STARTING_BATCH
    with open("weights_and_biases.txt", "r") as file:
        saved_data = json.load(file)
        w, b, STARTING_BATCH = saved_data # destructuring
    # initialize differentials and activations to 0
    for i in range(len(LAYERS)-1):
        dm = [[0 for _ in range(LAYERS[i])] for _ in range(LAYERS[i+1])]
        dm_sum = [[0 for _ in range(LAYERS[i])] for _ in range(LAYERS[i+1])]
        dw.append(dm)
        dw_sum.append(dm_sum)
    b = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))] # biases; start at 0
    db = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))] # dC / db
    db_sum = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))]
    act = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))] # activations
    da = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))] # dC / dact
    da_sum = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))]

def random_initialization():
    global w, dw, dw_sum, b, db, db_sum, act, da, da_sum, STARTING_BATCH
    # initializes w as randoms
    for i in range(len(LAYERS)-1):
        m = [[random.gauss(0, math.sqrt(2 / LAYERS[i])) for _ in range(LAYERS[i])] for _ in range(LAYERS[i+1])] # He initialization to avoid exploding activation in recursive ReLUs
        dm = [[0 for _ in range(LAYERS[i])] for _ in range(LAYERS[i+1])]
        dm_sum = [[0 for _ in range(LAYERS[i])] for _ in range(LAYERS[i+1])]
        w.append(m)
        dw.append(dm)
        dw_sum.append(dm_sum)
    # start biases and activations at 0
    b = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))] # biases; start at 0
    db = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))] # dC / db
    db_sum = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))]
    act = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))] # activations
    da = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))] # dC / dact
    da_sum = [[0 for _ in range(LAYERS[i])] for i in range(len(LAYERS))]

load_initialization() # toggle between random_initialization() and load_initialization()

def sig(x): 
    if (x <= -10):
        return 1
    return 1 / (1 + math.exp(-x))

def sig_p(x):
    if abs(x) > 16:
        return 0
    return math.exp(-x) / pow(1 + math.exp(-x), 2)

def relu(x):
    return max(0, x)

def relu_p(x):
    return 1 if x >= 0 else 0

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

questions, correct = 0, 0

def forward_propagation(image, label, id):
    global questions, correct
    clear_activations() # reset activations
    clear_differentials() # resets differentials
    assert(len(image) == 784)
    for i in range(LAYERS[0]):
        act[0][i] = image[i] / 255.0 # convert grayscale to float
    for i in range(len(LAYERS)-1):
        for j in range(LAYERS[i+1]): # compute next neuron's activation
            if i == N-2:
                act[i+1][j] = dot(act[i], w[i][j]) + b[i+1][j]
            else:
                act[i+1][j] = relu(dot(act[i], w[i][j]) + b[i+1][j])
    act[N-1] = softmax(act[N-1])
    C = 0 
    for i in range(LAYERS[N-1]):
        C += pow(act[N-1][i] - (1 if label == i else 0), 2)
    bestGuess = 0
    for i in range(10):
        if act[N-1][i] > act[N-1][bestGuess]:
            bestGuess = i
    questions += 1
    if bestGuess == label:
        correct += 1
    
    if id % BATCH_SIZE == 0:
        print(f"Cost for batch {id} = {C}, best guess is {bestGuess}, accuracy is {float(correct) / questions} ", end='')
        for i in range(10):
            print(act[N-1][i], end=' ')
        print()
    

BATCH_SIZE = 200

def softmax(a):
    max_a = max(a)
    exps = [math.exp(i - max_a) for i in a]
    exp_sum = sum(exps)
    return [i / exp_sum for i in exps]

def back_propagation(image, label, id):
    # start with last layer's dC / da
    da[N-1] = act[N-1]
    da[N-1][label] -= 1

    for i in range(N-2, -1, -1):
        for j in range(LAYERS[i+1]): 
            sum_within = dot(act[i], w[i][j]) + b[i+1][j]
            df = sig_p(sum_within) if i == N-2 else relu_p(sum_within)
            db[i+1][j] = da[i+1][j] * df # differentials for b
            for k in range(LAYERS[i]): # differentials for w and a
                dw[i][j][k] = da[i+1][j] * df * act[i][k]
                da[i][k] += da[i+1][j] * df * w[i][j][k]


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
    
STEP_SIZE = 0.05 # try first with constant step sizes

for id, data in enumerate(zip(train_x, train_y)):
    if (id <= STARTING_BATCH):
        continue
    image, label = data

    # every BATCH_SIZE, perform a mini-batch gradient descent
    if id % BATCH_SIZE == 0:
        gradient_descent(STEP_SIZE)
        clear_sums()
    
    forward_propagation(image, label, id)

    # every 2*BATCH_SIZE, print an image and see if the prediction is right
    if id % (BATCH_SIZE) == 0:
        for i in range(784):
            print("#" if image[i] > 125 else '.', end = "\n" if i % 28 == 27 else "")
        print(f"correct answer is {label}")

    # every 100, save the current weights and biases into a txt file
    if id % 100 == 0:
        cur_data = (w, b, id) # can start from training data (id+1) next time
        with open("weights_and_biases.txt", "w") as file:
            json.dump(cur_data, file, indent=4)
        
    back_propagation(image, label, id)
    contribute_to_sum()