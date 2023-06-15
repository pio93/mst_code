import numpy as np
from tensorflow import keras

# Shuffle data without affecting the original dataset
def shuffle_data(data):
    new_data = np.copy(data)
    np.random.shuffle(new_data)
    return new_data

class GAN:
    def __init__(self, epochs, batch_size):

        '''
        Arguments:
        epochs (int) : number of epochs for training
        batch_size (int) : size of a single batch
        '''

        self.epochs = epochs
        self.batch_size = batch_size

    def create_gun(self, X):
        sample_size = X.shape[1]
        self.input_size = sample_size // 2

        # Create generator
        kernel_initializer = keras.initializers.random_normal(stddev=0.01, seed=42)
        bias_initializer = keras.initializers.Zeros()

        self.generator = keras.models.Sequential()
        self.generator.add(keras.layers.Dense(128, input_shape=(self.input_size, ), 
                                              kernel_initializer=kernel_initializer, 
                                              bias_initializer=bias_initializer))
        self.generator.add(keras.layers.LeakyReLU(alpha=0.7))
        self.generator.add(keras.layers.BatchNormalization(momentum=0.9))
        self.generator.add(keras.layers.Dropout(0.2))
        self.generator.add(keras.layers.Dense(256))
        self.generator.add(keras.layers.LeakyReLU(alpha=0.7))
        self.generator.add(keras.layers.BatchNormalization(momentum=0.9))
        self.generator.add(keras.layers.Dropout(0.2))
        self.generator.add(keras.layers.Dense(512))
        self.generator.add(keras.layers.LeakyReLU(alpha=0.7))
        self.generator.add(keras.layers.BatchNormalization(momentum=0.9))
        self.generator.add(keras.layers.Dropout(0.2))
        self.generator.add(keras.layers.Dense(sample_size, activation='tanh'))

        # Create discriminator
        self.discriminator = keras.models.Sequential()
        self.discriminator.add(keras.layers.Dense(512, input_shape=(sample_size, ), 
                                                  kernel_initializer=kernel_initializer, 
                                                  bias_initializer=bias_initializer))
        self.discriminator.add(keras.layers.LeakyReLU(alpha=0.7))
        self.discriminator.add(keras.layers.BatchNormalization(momentum=0.9))        
        self.discriminator.add(keras.layers.Dropout(0.2))
        self.discriminator.add(keras.layers.Dense(256))
        self.discriminator.add(keras.layers.LeakyReLU(alpha=0.7))
        self.discriminator.add(keras.layers.BatchNormalization(momentum=0.9))        
        self.discriminator.add(keras.layers.Dropout(0.2))
        self.discriminator.add(keras.layers.Dense(128))
        self.discriminator.add(keras.layers.LeakyReLU(alpha=0.7))
        self.discriminator.add(keras.layers.BatchNormalization(momentum=0.9))        
        self.discriminator.add(keras.layers.Dropout(0.2))
        self.discriminator.add(keras.layers.Dense(1, activation='sigmoid'))

        optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.4)

        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.discriminator.trainable = False

        self.gan = keras.models.Sequential()
        self.gan.add(self.generator)
        self.gan.add(self.discriminator)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)


    def train_gun(self, X, y):
        minority_data = np.take(X, np.where(y == 1)[0], axis=0)

        num_batches = minority_data.shape[0] // self.batch_size 

        if num_batches == 0: num_batches = 1

        for e in range(self.epochs):
            rand_min_data = shuffle_data(minority_data)
            start = 0
            disc_loss_arr = []
            gen_loss_arr = []

            for _ in range(num_batches):
                real_data = rand_min_data[start: start + self.batch_size]
                noise = np.random.uniform(0, 1, (real_data.shape[0], self.input_size))
                gen_data = self.generator(noise)
                train_data = np.vstack((gen_data, real_data))
                train_labels = np.array([0] * real_data.shape[0] + [1] * real_data.shape[0])
                self.discriminator.trainable = True
                disc_loss = self.discriminator.train_on_batch(train_data, train_labels)
                self.discriminator.trainable = False
                disc_loss_arr.append(disc_loss)
                noise = np.random.normal(0, 1, (real_data.shape[0], self.input_size))
                labels = np.array([1] * real_data.shape[0])
                gen_loss = self.gan.train_on_batch(noise, labels)
                gen_loss_arr.append(gen_loss)
                start = start + self.batch_size
            
            print('Epoch: {}\tdiscriminator loss: {}\tgenerator loss: {}'.format(e, np.mean(disc_loss_arr), np.mean(gen_loss_arr)))

    def generate_samples(self, X, y):
        majority = np.take(X, np.where(y != 1)[0], axis=0)
        minority = np.take(X, np.where(y == 1)[0], axis=0)

        rate = majority.shape[0] - minority.shape[0]
        synthetic_data = []
        synthetic_labels = []

        for _ in range(rate):
            noise = np.random.uniform(0, 1, (1, self.input_size))
            new_sample = self.generator(noise)
            synthetic_data.append(new_sample[0])
            synthetic_labels.append(1)

        synthetic_data = np.array(synthetic_data)
        synthetic_labels = np.array(synthetic_labels)
        new_data = np.vstack((X, synthetic_data))
        new_labels = np.hstack((y, synthetic_labels))

        return new_data, new_labels

    def fit_resample(self, X, y):

        self.create_gun(X)

        self.train_gun(X, y)

        new_X, new_y = self.generate_samples(X, y)

        return new_X, new_y
    



