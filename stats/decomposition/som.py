# -*- coding: utf-8 -*-

"""
@file som.py
@brief SOM (Self-Organizing Map)
@author ふぇいと (@stfate)

@description

"""

import scipy as sp
import scipy.linalg as linalg


def som( data , nx, ny, plotLabels = None):
    '''
        Implemented by Maezawa-san
        
        自己組織化マップで、高次元データを２次元に圧縮する
        ２次元のデータとは、XとY軸から構成されるデータであり、
        XとYはグリッド上で表現される。
        
        [引数]
         data - N x D 次元のデータ。Nはデータの総数、Dはデータの次元
         nx - X軸の分割数
         ny - Y軸の分割数
         plotLabels - None以外の場合、データをプロットする、この場合、plotLabelsはN次元の文字列配列である必要がある。
        [戻り値]
         X, Y, IDX のTuple
         X - X軸のリスト
         Y - Y軸のリスト
         IDX - N次元の配列。
        　　　　　　　各入力データ点が、X軸とY軸のどこに対応するかが保存されている。 
        　　　　　　　data[n,:]に対応するデータは X[ IDX[n] ], Y[IDX[n]] に射影されている。
        
    '''
    X,Y = sp.meshgrid( sp.arange(nx), sp.arange(ny) )
    X = X.flatten()
    Y = Y.flatten()
    nData, nDim = data.shape
    IDX = sp.zeros( nData )
    weightVec = sp.zeros( (len(X),nDim ) )
    for i in xrange( len(X) ):
        weightVec[i,:] = data[ i%nData, : ]
    weightVec += sp.randn(  len(X), nDim) * data.std(0)[sp.newaxis,:] * 0.01 #+ data.mean(0)[sp.newaxis,:]
    
    # compute the distace
    alpha = 1.0 #0.99999
    r = float( nx*nx+ny*ny  )
    iPrev=sp.zeros_like(IDX)
    for it in xrange(100):
        # print it
        for i in xrange(nData):
            diff = data[i,:] - weightVec[:,:]
            distance = (diff*diff).sum(1)
            bmu = distance.argmin()
            IDX[i] = bmu
            T = sp.exp( - 0.5* ( (X-X[bmu])*(X-X[bmu]) + (Y-Y[bmu])*(Y-Y[bmu]) ) / r )
            #print T
            weightVec += T[:,sp.newaxis]*alpha*diff[sp.newaxis,bmu,:] #+ sp.randn( weightVec.shape[0], weightVec.shape[1] )*0.1
            # print weightVec
        
        #exit()
        #if ( (iPrev==IDX).all() ):break
        #iPrev[:] = IDX[:]
        alpha *= 0.98
        r *= 0.91
        
        if (plotLabels):
            print IDX.astype('int')
            pp.clf()
            for i in xrange( len(IDX)) :
                #print IDX[i],X[IDX[i]], Y[IDX[i]]
                pp.text( sp.randn(1)*0.01 + X[IDX[i]], sp.randn(1)*0.01 + Y[IDX[i]], plotLabels[i] )
            pp.xlim( X.min(),X.max())
            pp.ylim( Y.min(), Y.max() )
            pp.title('a={0}; r= {1}'.format(alpha,sp.sqrt(r)) )
            pp.savefig('data/img/som.{0}.png'.format(it))
    for i in xrange(nData):
        d = sp.sqrt( ((weightVec-data[i,:])*(weightVec-data[i,:])).sum(1) )
        IDX[i] = d.argmin()
        # print IDX[i]
    # print IDX
    
    return X,Y,IDX
