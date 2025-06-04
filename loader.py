import struct

class Loader(object):
    def __init__(self, training_images_path, training_labels_path, test_images_path, test_labels_path):
        self.training_images_path = training_images_path
        self.training_labels_path = training_labels_path
        self.test_images_path = test_images_path
        self.test_labels_path = test_labels_path

    def packing_pairs(self, images_path, labels_path):
        labels = []
        with open(labels_path, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049: # magic number 2049 indicates labels
                raise ValueError("Magic number for labels is not 2049")
            labels = list(file.read(size))
        images = []
        with open(images_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051: # magic number 2051 indicates images
                raise ValueError("Magic number for images is not 2051")
            image_data = list(file.read(size * rows * cols))
        for i in range(size):
            images.append(list(image_data[i * rows * cols : (i+1) * rows * cols]))
        return images, labels
