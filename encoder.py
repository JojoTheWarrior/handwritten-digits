import json
import numpy as np

LAYERS = [784, 100, 10]
w_code, b_code = "", ""

def float_to_16_bits(x):
    x = np.float16(x)
    bits = format(np.frombuffer(x.tobytes(), dtype='uint16')[0], '016b')
    return bits

def read_list_data():
    with open("weights_rounded.txt", "r") as file:
        saved_data = json.load(file)
        w, b, _ = saved_data
        # stringing w together
        for i in range(len(LAYERS)-1):
            for j in range(LAYERS[i+1]):
                for k in range(LAYERS[i]):
                    w_code += float_to_16_bits(w[i][j][k])
        # stringing b together
        for i in range(1, len(LAYERS)):
            for j in range(LAYERS[i]):
                b_code += float_to_16_bits(b[i][j])

    with open("binary_weights_and_biases.txt", "w") as file:
        file.write(w_code + "\n" + b_code)

def string_to_unicode():
    with open("binary_weights_and_biases.txt", "r") as file:
        text = file.read()
    
    while len(text) % 20 != 0:
        text += "0"
    # grouping into 20s
    chars = []
    for i in range(0, len(text), 20):
        chunk = text[i:i+20]
        val = int(chunk, 2)

        if val > 0x10FFFF:
            raise ValueError(f"this unicode chunk of 21 bits starting from bit {i} exceeds the max unicode size")
        chars.append(chr(val))
    # printing into another file
    print(chars)
    with open("chinese.txt", "w", encoding="utf-16") as file:
        file.write(''.join(chars))

string_to_unicode()

