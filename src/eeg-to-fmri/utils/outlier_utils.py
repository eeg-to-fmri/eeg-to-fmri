import numpy as np


class IQR:
    def __init__(self, l=25, h=75):
        self.l=l
        self.h=h
        pass
    
    def fit(self, X):
        Q1 = np.percentile(X, self.l, interpolation = 'midpoint')
        Q3 = np.percentile(X, self.h, interpolation = 'midpoint')
        
        IQR = Q3 - Q1  
        
        self.low = Q1 - 1.5 * IQR
        self.up = Q3 + 1.5 * IQR
    
    def transform(self, X, channels_last=True):
        _X = np.copy(X)
        
        iqr_X=np.zeros(X.shape)

        self.outlier_values=[]
        self.outlier_idx=[]

        if(channels_last):
            for volume in range(_X.shape[3]):
                for x in range(_X.shape[0]):
                    for y in range(_X.shape[1]):
                        for z in range(_X.shape[2]):
                            if (np.any(_X[x,y,z,volume] > self.up) or np.any(_X[x,y,z,volume] < self.low)):
                                self.outlier_idx.append((x,y,z))
                                self.outlier_values.append(_X[x,y,z,volume])
                                _X[x,y,z,volume]=np.zeros(_X[x,y,z,volume].shape)
        else:
            for volume in range(_X.shape[0]):
                for x in range(_X.shape[1]):
                    for y in range(_X.shape[2]):
                        for z in range(_X.shape[3]):
                            if (np.any(_X[volume,x,y,z] > self.up) or np.any(_X[volume,x,y,z] < self.low)):
                                self.outlier_idx.append((x,y,z))
                                self.outlier_values.append(_X[volume,x,y,z])
                                _X[volume,x,y,z]=np.zeros(_X[volume,x,y,z].shape)
        
        return _X
    
    def inverse_transform(self, X, channels_last=True):
        _X = np.copy(X)
        
        for idx in range(len(self.outlier_idx)):
            if(channels_last):
                _X[self.outlier_idx[idx][0],self.outlier_idx[idx][1],self.outlier_idx[idx][2],:] = self.outlier_values[idx]
            else:
                _X[:,self.outlier_idx[idx][0],self.outlier_idx[idx][1],self.outlier_idx[idx][2]] = self.outlier_values[idx]
        
        return _X