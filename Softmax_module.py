#Hasnaa Bakri
import numpy as np
from sklearn.preprocessing import OneHotEncoder
class Softmax:
    def __init__(self , iterations=1000 , alpha=0.1):
        self.iterations=iterations
        self.alpha=alpha
        self.betas=None
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z,keepdims = True,axis=1))
        return exp_z / np.sum(exp_z,keepdims = True,axis=1)
    def gradient_descent(self , X , y ):
        num_samples , num_features=X.shape
        x_biased=np.c_[np.ones(num_samples) , X]
        num_classes = y.shape[1]
        self.betas=np.zeros((num_features+1 , num_classes ))
        for i in range(self.iterations):
            scores = np.dot(x_biased,self.betas)
            probs = self.softmax(scores)
            gradient = np.dot(x_biased.T, (probs -y )) /num_samples
            self.betas -= (self.alpha) * gradient
    def fit (self , X , y):
        num_classes = np.unique(y).shape[0]
        encoder = OneHotEncoder(sparse_output=False)
        y_encoded = encoder.fit_transform(y.reshape(-1, 1))
        self.gradient_descent(X , y_encoded)
    def predict(self , X):
        num_samples=X.shape[0]
        x_biased=np.c_[np.ones(num_samples) , X]
        return np.argmax(self.softmax(np.dot(x_biased , self.betas)) , axis=1)
    def score(self , X ,y):
        return np.mean(self.predict (X)==y)


        


