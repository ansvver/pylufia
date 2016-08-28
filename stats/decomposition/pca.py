# -*- coding: utf-8 -*-

"""
pca.py

PCAの実装
"""

import scipy as sp
import scipy.linalg as linalg


class PCA():
    """
    主成分分析
    """
    def __init__(self, n_components=2):
        self.n_components = n_components

    def train(self, X):
        # データの中心化
        self.X_mean = X.mean(0)
        X_centered = X - self.X_mean

        # 分散共分散行列の作成
        V = sp.cov(X_centered.T)
        
        # Vの固有値計算
        self.eigvals,self.eigvecs = linalg.eig(V)
        
        # 大きい方からn_components個の固有値を取り出し，それに対応する固有ベクトルを並べて基底を定める
        eigvals_idx = sp.argsort(self.eigvals)
        eigvals_idx = eigvals_idx[len(eigvals_idx)::-1]
        self.U = self.eigvecs[eigvals_idx[:self.n_components]]
        
        # 基底ベクトルから射影した点を求める
        X_pca = sp.dot(self.U, X_centered.T)
        X_pca = X_pca.T

        return X_pca, self.U

    def predict(self, X):
        X_centered = X - self.X_mean
        X_pca = sp.dot(self.U, X_centered.T)
        X_pca = X_pca.T

        return X_pca
