import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf


image_size = 8

def create_result(ball_x, bat_x):
    ball_position = ball_x - bat_x
    left = ball_position < 0
    hit = (ball_position >= 0) & (ball_position < 2)
    right = ball_position >= 3
    return (left, hit, right)



def create_image(image_size, show_image=False):
    ball_x = np.random.randint(0, image_size)
    ball_y = np.random.randint(0, image_size-1)
    bat_x = np.random.randint(0,5)
    image = np.zeros((image_size, image_size))
    image[ball_y, ball_x] = 256
    image[image_size - 1, bat_x:bat_x+3] = 256
    if show_image:
        plt.imshow(image, cmap='Greys_r')
        print(create_result(ball_x, bat_x))
        plt.show()
    out = create_result(ball_x, bat_x)
    return image, out

data_set_size = 200000
X = np.zeros((data_set_size, image_size, image_size))
Y = np.zeros((data_set_size, 3))


def create_data_set(data_set_size, image_size):
    for i in range(data_set_size):
        image, out = create_image(image_size)
        X[i,...] = image
        Y[i,:] = out
    return X, Y

if __name__ == '__main__':
    X, Y = create_data_set(data_set_size, image_size)
    pickle.dump(X, open('data/X.p', 'wb'))
    pickle.dump(Y, open('data/Y.p', 'wb'))

    Xl = pickle.load(open('data/X.p', 'rb'))
    Yl = pickle.load(open('data/Y.p', 'rb'))

    print(Xl)
    print(Yl)

