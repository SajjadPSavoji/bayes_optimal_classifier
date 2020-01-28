import numpy as np
import operator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import copy
from mpl_toolkits import mplot3d

NOT_IMPLEMENTED_ERROR = "NOT_IMPLEMENTED_ERROR"

def euclidean_distance(x1 , x2):
    return np.linalg.norm(x1 - x2)

class Window():
    def __init__(self , h , n):
        self.h = h
        self.n = n
        self.scale = np.power(1/h , self.n)

    def clip(self , x , x0 , h = None):
        raise NOT_IMPLEMENTED_ERROR

class Rect_window(Window):
    def __init__(self , h , n):
        super().__init__(h , n)
    
    def clip(self , x , x0 , h = None):
        if h is None : h = self.h

        v = abs((x0 -x)/h )

        if sum(v <= 1) == self.n:
            return   self.scale
        return 0

class Gaussian_window(Window):
    def __init__(self , h , n):
        super().__init__(h , n)
        self.my_scalse = 1 / np.power(2*np.pi , self.n/2)

    def clip(self , x , x0 , h = None):
        if h is None : h = self.h

        v = (x0 - x)/h

        return self.scale * self.my_scalse * np.exp(-0.5*np.dot(v , v))



    

class Dist():
    def __init__(self):
        raise NOT_IMPLEMENTED_ERROR

    def estimate(self , data):
        raise NOT_IMPLEMENTED_ERROR

    def disc_func(self , x):
        raise NOT_IMPLEMENTED_ERROR

    def p(self , x):
        raise NOT_IMPLEMENTED_ERROR

class gaussian_dist(Dist):
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.scale = None
        self.isigma = None
        self.n = None
    
    def estimate(self , data):
        # data size of this class
        self.Q = data.shape[0]
        self.n = data.shape[1]

        # estimate mu
        self.estimate_mu(data)

        # estimate sigma
        self.estimate_simga(data)

        # feature conditioning
        self.feature_condition()

        # update params ie. sigma and mu
        self.condition_params()

        # compute inverse of covarience matrix
        self.isigma = np.linalg.inv(self.sigma)

        # compute constant numbers in disc function to boost performance
        self.det = abs(np.linalg.det(self.sigma))
        self.ln_det_sigma = np.log(self.det)
        self.scale = 1/np.power(self.det , 0.5)

    def estimate_mu(self , data):
        # initilize mu
        d = np.atleast_2d(data[0])
        d = d.T
        self.mu = np.zeros(d.shape)

        # compute mu
        for sample in data :
            d = np.atleast_2d(sample)
            d = d.T
            self.mu += d
        self.mu /= self.Q


    def estimate_simga(self , data):
        # initilize mu and sigma with zeros
        d = np.atleast_2d(data[0])
        d = d.T
        self.sigma = np.zeros((np.matmul(d , d.T)).shape)

        # compute sigma
        for sample in data :
            d = np.atleast_2d(sample)
            d = d.T
            self.sigma += np.matmul(d-self.mu , (d-self.mu).T)
        self.sigma /= self.Q

    def feature_condition(self):
        self.A = np.eye(self.n)
        self.A = np.matmul(self.feature_selection() , self.A)

    def condition_params(self):
        self.mu = np.matmul(self.A , self.mu)
        self.sigma = np.matmul(np.matmul(self.A , self.sigma) , self.A.T)

    def dims_to_select(self):
        '''
            should be called after sigma is computed
        '''
        selected = []
        for dim in range(self.n):
            if not self.sigma[dim][dim] == 0:
                selected.append(dim)
        return selected


    def feature_selection(self):
        dims_selected = self.dims_to_select()
        m = len(dims_selected)
        A = np.zeros((m , self.n))
        for i , dim in enumerate(dims_selected):
            A[i][dim] = 1
        
        return A

    def disc_func(self , x):
        return self.p(x)

        # d = np.atleast_2d(x)
        # d = d.T
        # d = np.matmul(self.A , d)
        # return self.ln_det_sigma  + np.matmul(np.matmul((d-self.mu).T , self.isigma) , (d-self.mu))

    def p(self , x):
        d = np.atleast_2d(x)
        d = d.T
        d = np.matmul(self.A , d)
        return self.scale * np.exp(-1/2 * np.matmul(np.matmul((d-self.mu).T , self.isigma) , (d-self.mu)))

class parzen_estimate(Dist):
    def __init__(self ,window):
        self.window = window

    def estimate(self , data):
        self.data = data
        self.Q = data.shape[0]

    def p(self , x0):
        sum = 0
        for sample in self.data :
            sum += self.window.clip(sample , x0)
        return sum / self.Q

    def disc_func(self , x0):
        return self.p(x0)

class knn_estimate(Dist):
    def __init__(self , k=1):
        self.k = k

    def estimate(self , data):
        self.data = data
        self.Q = data.shape[0]
        self.n = len(data[0])

    def p(self , x0):
        d = copy.deepcopy(self.data)
        dists = np.linalg.norm(d - x0 ,axis=1)
        idx = np.argpartition(dists , self.k)
        far_feature= self.data[idx[self.k]]
        
        return self.k /self.Q/ np.power(np.linalg.norm(x0 - far_feature) , self.n)





class Data_set():
    def __init__(self):
        pass
    def init_risk(self , risk_mat):
        '''
        risk matrix is a matrix denoted with L 
        l[j][i] is denoted to risk of choozing j while data belongs to class i
        '''
        self.l = risk_mat

    def fit(self , features = None , labels = None , distribution = gaussian_dist()):
        '''
        either load should be called fisrt or fit should be called with appropriate features and labels
        '''
        if features is None and labels is None:
            self.compute_priors()
            # self.normalize()
            self.estimate_distributions(dist= distribution)
            return

        else :
            if features is not None : self.features = features
            if labels   is not None : self.labels = labels
            self.fit()

    def load(self , file_name):
        features = []
        labels = []
        with open(file_name) as fd:
            for line in fd:
                features.append(list(map(int , line.split(',')))[0:-1])
                labels.append(list(map(int , line.split(',')))[-1])

            self.features = np.array(features)
            self.labels = np.array(labels)
            self.max_feature_vec = np.max(self.features , axis=0)
            self.data_size = len(labels)

    def compute_priors(self):

        self.priors = {}
        for label in self.labels :
            if label in self.priors:
                self.priors[label] += 1
            else :
                self.priors[label] = 1

        for key in self.priors.keys():
            self.priors[key] /= self.data_size

        self.c = len(self.priors)   

    def normalize(self , data = None):
        if data is None:
            self.features = self.features / self.max_feature_vec
            return None
        return data / self.max_feature_vec


    def estimate_distributions(self , dist):

        self.distributions = {}

        for key in self.priors.keys():
            self.distributions[key] = copy.deepcopy(dist)

        for key in self.distributions :
            self.distributions[key].estimate(self.features[self.labels == key])
    

    def transform(self , new_instance):
        discs = {}
        for key in self.distributions.keys() :
            discs[key] = self.distributions[key].p(new_instance) * self.priors[key]

        return max(discs.items(), key=operator.itemgetter(1))[0]

    def classify_all(self , new_mat_instace):
        labels = np.zeros((new_mat_instace.shape[0] , ))
        for i , instance in enumerate(new_mat_instace) :
            labels[i] = self.transform(instance)
        
        return labels

    def transform_risk(self  ,new_instance):
        discs = {}
        # init risk values
        for key in self.distributions.keys() :
            discs[key] = 0

        r_key  = self.c
        discs[r_key] = 0

        # print("reject key: " , self.c)

        # compute risk for j
        for decision in discs.keys():
            for clss in self.distributions.keys():
                discs[decision] += self.l[decision][clss] * self.distributions[clss].p(new_instance) * self.priors[clss]

        return min(discs.items(), key=operator.itemgetter(1))[0]

    def classify_all_risk(self , new_mat_instace):
        labels = np.zeros((new_mat_instace.shape[0] , ))
        for i , instance in enumerate(new_mat_instace) :
            labels[i] = self.transform_risk(instance)
        
        return labels

# report functions

def acc(true_labels , pred_labels):
    return accuracy_score(true_labels , pred_labels)

def ccr(true_labels , pred_labels , reject_label):
    num_reject = sum(pred_labels == reject_label)
    num_true = sum(pred_labels == true_labels)
    if num_reject == len(true_labels): return 0
    return num_true / (len(true_labels) - num_reject)


def confusion_mat(true_labels , pred_labels):
    return confusion_matrix(true_labels , pred_labels).T

def normalize_conf_mat(conf_mat):
    norm = np.sum(conf_mat , axis=0)
    return conf_mat / norm

def plot_confusion_mat(conf_mat):
    plt.matshow(conf_mat , cmap = plt.cm.gray)
    plt.colorbar()

def risk_matrix(s , r , c):
    '''
    c is the number of classes 
    s is the risk associated with risk of normal errors
    r is the risk associated with rejection 
    '''

    L = np.ones((c+1 , c+1)) - np.eye(c+1)
    L = L * s
    
    last_row = np.array([r for _ in range(c+1)])
    L[c] = last_row

    return L

def exhastive_search(train , test , low = 0 , high = 1 , step = 0.1 , result = None):
    stat = {}
    for s in tqdm_notebook(np.arange(low , high , step)):
        for r in np.arange(low , high , step):
            train.init_risk(risk_matrix(s , r, train.c))
            pred = train.classify_all_risk(test.features)
            ac = ccr(test.labels , pred , train.c)
            stat[(s , r)] = ac

    if result is None:
        key = max(stat.items(), key=operator.itemgetter(1))[0]  
        return key , stat[key]
    return stat

def exhastive_search_for_h(train , test , low = 100 , high = 200 , step = 10 , result = None , win = Window):
    stat = {}
    for h in tqdm_notebook(np.arange(low , high , step)):
        my_win = win(h, 16)
        dist= parzen_estimate(my_win)
        train.fit(distribution=dist)
        pred = train.classify_all(test.features)
        ac =  acc(test.labels , pred)
        stat[h] = ac

    if result is None:
        key = max(stat.items(), key=operator.itemgetter(1))[0]  
        return key , stat[key]  
    return stat

def exhastive_search_for_k(train , test , low = 1 , high = 100 , step = 1 , result = None):
    stat = {}
    for k1 in tqdm_notebook(np.arange(low , high , step)):
        dist = knn_estimate(k = k1)
        train.fit(distribution=dist)
        pred = train.classify_all(test.features)
        ac =  acc(test.labels , pred)
        stat[k1] = ac

    if result is None:
        key = max(stat.items(), key=operator.itemgetter(1))[0]  
        return key , stat[key]  
    return stat




class knn_classifier():
    def __init__(self , k = 1):
        self.k = k

    def fit(self , data , labels):
        self.data = data
        self.labels = labels

    def transform(self , query):
        pred = np.zeros((query.shape[0] , ))
        for i , q1 in enumerate(query): 
            pred[i] = self.transform1(q1)

        return pred

        

    def fit_transform(self , data , labels , query):
        self.fit(data , labels)
        self.transform(query)

    def transform1(self , q1):
        d = copy.deepcopy(self.data)
        d_mat = d - q1
        distances = np.linalg.norm(d_mat , axis=1)
        idx = np.argpartition(distances , self.k)
        knn_labels = self.labels[idx[:self.k]]
        return np.bincount(knn_labels).argmax()



if __name__ == "__main__":
    pass

    