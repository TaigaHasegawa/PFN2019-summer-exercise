
import numpy as np

#function to make Adjacency matrix
def make_symmetric_matrix(arr,shape):
    """
    arr: arr which specify where is non-zero
    shape: shape of the symmetric matrix
    """
    G=np.zeros((shape))
    for i in arr:
        G[i[0]][i[1]]=1
        G[i[1]][i[0]]=G[i[0]][i[1]]
    return G

#relu function 
def relu(x):
    y = np.maximum(0, x)
    return y

#return feature vector h
def GNN_basic(G,X,t):
    """
    G: Adjacency matrix
    X: feature vector 
    t: the number of steps
    """
    for i in range(t):
        a=np.dot(G,X)
        X=relu(np.dot(a,W))
    h=np.sum(X,axis=0)
    return h

#Adjacency matrix for test
G=make_symmetric_matrix(np.array([[1,0],[2,1],[3,2],[3,4],[1,3],[4,2],[4,5],[6,4],[6,3],[6,5],[7,2],[7,0],[7,3],[8,0],[8,5],[8,6],[9,2],[9,8]]),(10,10))
#feature vector when D=4
X=np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]])

#dimension of feature vector 
D=4
#set the seed
np.random.seed(2018)
#initialize D*D size weight parameter 
W=np.random.normal(0,0.4,(D,D))

#toy example
h=GNN_basic(G,X,t=2)

#show the result of toy example 
if __name__ == "__main__":
    print(h)
#array([ 0., 31.56446968, 22.3687154 ,  0.])