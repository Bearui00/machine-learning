# coding=utf-8
import numpy as np
from keras.datasets import mnist
from layers import FullyConnectedLayer, SigmoidLayer, SoftmaxLossLayer
import time

(x_train,y_train),(x_test,y_test)=mnist.load_data()
data_train=[]
label_train=[]
data_test=[]
label_test=[]
for i in range(x_train.shape[0]):
    if y_train[i]<=2:
        data_train.append([x for y in x_train[i] for x in y])
        label_train.append([y_train[i]])
for i in range(x_test.shape[0]):
    if y_test[i]<=2:
        data_test.append([x for y in x_test[i] for x in y])
        label_test.append([y_test[i]])
data_train=np.array(data_train)
label_train=np.array(label_train)
data_test=np.array(data_test)
label_test=np.array(label_test)

# h1 h2 lr     ac
# 32 16 0.01   64
# 32 16 0.001  98.5

class MNIST_MLP(object):
    def __init__(self, input_size=784, hidden1=32, hidden2=16, out_classes=3, lr=0.001, max_epoch=3):
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch


    def load_data(self):
        print('Loading MNIST data from files...')
        train_images = data_train
        train_labels = label_train
        test_images = data_test
        test_labels = label_test
        self.train_data = np.append(train_images, train_labels, axis=1)
        self.test_data = np.append(test_images, test_labels, axis=1)

    def shuffle_data(self):
        print('Randomly shuffle MNIST data...')
        np.random.shuffle(self.train_data)

    def build_model(self):  # 建立网络结构
        print('Building multi-layer perception model...')
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.sig1 = SigmoidLayer()
        self.fc2=FullyConnectedLayer(self.hidden1, self.hidden2)
        self.sig2 = SigmoidLayer()
        self.fc3 = FullyConnectedLayer(self.hidden2, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3]

    def init_model(self):
        print('Initializing parameters of each layer in MLP...')
        for layer in self.update_layer_list:
            layer.init_param()

    def forward(self, input):  # 神经网络的前向传播
        h1 = self.fc1.forward(input)
        h1 = self.sig1.forward(h1)
        h2=self.fc2.forward(h1)
        h2=self.sig2.forward(h2)
        h3=self.fc3.forward(h2)
        prob = self.softmax.forward(h3)
        return prob

    def backward(self):  # 神经网络的反向传播
        dloss = self.softmax.backward()
        dh3=self.fc3.backward(dloss)
        dh3=self.sig2.backward(dh3)
        dh2=self.fc2.backward(dh3)
        dh1 = self.sig1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def train(self):
        max_batch = self.train_data.shape[0]
        print('Start training...')
        for idx_epoch in range(self.max_epoch):
            self.shuffle_data()
            for idx_batch in range(int(max_batch)):
                batch_images = self.train_data[idx_batch:(idx_batch+1), :-1]
                batch_labels = self.train_data[idx_batch:(idx_batch+1), -1]
                prob = self.forward(batch_images)
                loss = self.softmax.get_loss(batch_labels)
                self.backward()
                self.update(self.lr)
                # if idx_batch % self.print_iter == 0:
                #     print('Epoch %d, iter %d, loss: %.6f' % (idx_epoch, idx_batch, loss))

    def evaluate(self):
        pred_results = np.zeros([self.train_data.shape[0]])
        for idx in range(int(self.train_data.shape[0])):
            batch_images = self.train_data[idx:(idx+1), :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            # print(self.train_data[idx:(idx+1),-1],pred_labels,idx)
            pred_results[idx:(idx+1)] = pred_labels
        accuracy = np.mean(pred_results == self.train_data[:,-1])
        print('Accuracy in train set: %f' % accuracy)

        pred_results = np.zeros([self.test_data.shape[0]])
        for idx in range(int(self.test_data.shape[0])):
            batch_images = self.test_data[idx:(idx+1), :-1]
            prob = self.forward(batch_images)
            pred_labels = np.argmax(prob, axis=1)
            # print(self.test_data[idx:(idx+1),-1],pred_labels,idx)
            pred_results[idx:(idx+1)] = pred_labels
        accuracy = np.mean(pred_results == self.test_data[:,-1])
        print('Accuracy in test set: %f' % accuracy)

def build_mnist_mlp(param_dir='weight.npy'):
    h1, h2, e = 32, 16, 3
    mlp = MNIST_MLP(hidden1=h1, hidden2=h2, max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    start=time.time()
    mlp.train()
    end=time.time()
    t=end-start
    print("Training Time = %f " % t )
    return mlp

if __name__ == '__main__':
    mlp = build_mnist_mlp()
    mlp.evaluate()
