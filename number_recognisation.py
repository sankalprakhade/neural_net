import numpy as np
import random

class Network (object):

    def __init__(self,sizes):
        self.no_of_layer = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        self.weight = [np.random.randn(y,x) for x,y in zip(sizes[0:-1],sizes[1:])]

    def feednextlayer(self,a):
        for w,b in (self.weight, self.biases):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def grad_d(self,training_data, epoch, mini_size, alpha, test_data):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range (epoch):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_size] for k in range(0,n,mini_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,alpha)
            if test_data:
                print('epoch{0}: {1} of {2}'.format(j,self.evaluate(test_data),n_test))
            else:
                print('epoch {0} completed'.format(j))
    def update_mini_batch(self,mini_batch,alpha):
        sum_diff_b = [np.zeros(b.shape) for b in self.biases]
        sum_diff_w = [np.zeros(w.shape) for w in self.weight]
        for x,y in mini_batch:
            diff_b, diff_w = self.backprop(x,y)
            sum_diff_b = [nb+db for nb,db in zip(sum_diff_b,diff_b)]
            sum_diff_w = [nw+dw for nw,dw in zip(sum_diff_w,diff_w)]
        self.weight = [w-alpha/len(mini_batch)*sdw for w,sdw in zip(self.weight,sum_diff_w)]
        self.biases = [b-alpha/len(mini_batch)*sdb for b,sdb in zip(self.biases,sum_diff_b)]

    def backprop(self,x,y):
        diff_b = [np.zeros(b.shape) for b in self.biases]
        diff_w = [np.zeros(w.shape) for w in self.weight]

        a = x
        a_s = [x]
        zs = []

        for b,w in zip(self.biases,self.weight):
            z = np.dot(w,a) + b
            zs = zs.append(z)
            a = sigmoid(z)
            a_s = a_s.append(a)

        dl = (a_s[-1]-y)*sigmoid_p(zs[-1])
        diff_b[-1] = dl
        diff_w[-1] = np.dot(dl,a[-2].transpose())

        for i in range(2,self.no_of_layer):
            sigmoid_d = sigmoid_p(zs[-i])
            dl = np.dot(self.weight[-i+1].transpose(),dl)*sigmoid_d
            diff_b[-i] = dl
            diff_w[-i] = np.dot(dl,a_s[-(i+1)].transpose())
        return diff_b,diff_w

    def predict(self,test_data):
        results = [np.argmax(self.feednextlayer(x)) for x in test_data[:][0]]
        return results

    def evaluate(self,test_data):
        return sum(self.predict(test_data)==test_data[:][1])

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_p(z):
    return sigmoid(z)*(1-sigmoid(z))
