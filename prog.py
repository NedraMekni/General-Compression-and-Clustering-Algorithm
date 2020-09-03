from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import datasets, cluster
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm
from  scipy.stats import norm
from scipy.cluster.hierarchy import dendrogram, linkage
from kneed import DataGenerator, KneeLocator
from rdkit import Chem

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import matplotlib
import random
import rdkit.Chem.Descriptors

dimension = 0 # number of descriptors, will be automatically computed in csv-smi parser
transf = lambda i: (i+1)*0.5
colors = []

dict_names = {}
title = []
cmp_name = []
raw_value = []
fname = "9.cls.smi"
def rand_col(min_v=0.5, max_v=1.0):
  hsv = np.concatenate([np.random.rand(2), np.random.uniform(min_v, max_v, size=1)])
  return matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb(hsv))


def parse_csv(name):
  global dimension,raw_value,cmp_name
  i = 0
  with open(fname,'r') as f:
    title = f.readline().strip().split(',')[1:]
    dimension=len(title)
    for l in f:
      cmp_name += [l.strip().split(',')[0][1:-1]]
      raw_value += [[float(x) if len(x)>0 else -1.7976931348623157e+308 for x in l.strip().split(',')[1:]]]


def labeling_results(res,plt,labels):
  global dict_names
  for i, txt in enumerate(res):    
    print('{} => {}'.format(i,dict_names[i]))
    plt.annotate(i, (txt[0],txt[1]))
  return plt


def runPCA(dsc_list, n_comp):
  pca = PCA(n_components=n_comp)
  crds = pca.fit_transform(dsc_list)
  return crds

# clst_type means cluster type e.g. (0)DBSCAN or (1)Hierarchical

def do_clustering(clst_type, data):
  if clst_type == 0:
    return DBSCAN(min_samples=2,eps=8.5).fit(data)

  if clst_type == 1:
    dend = dendrogram(linkage(data,method='ward'))
    plt.show()
    return AgglomerativeClustering(n_clusters=int(input('Numero di cluster in uscita: '))).fit(data)
  print("* clst_type not set properly")
  exit()  

def clusters_evaluation(clustering,data):
  cluster_labels= clustering.fit_predict(data)
  silhouette_avg = silhouette_score(data, cluster_labels)
  return silhouette_avg

def parse_smi(fname):
  global raw_value,cmp_name,dimension
  with open(fname,'r') as f:
    for l in tqdm(f.readlines()):
      cmp_name+=[l.strip()]
      m1 = Chem.MolFromSmiles(l.strip())
      raw_value+=[[eval("__import__('rdkit').Chem.Descriptors."+y+'(m1)',{'m1':m1}) for y in [x for x in Chem.Descriptors.__dict__ if x[0]!='_' and eval("type(__import__('rdkit').Chem.Descriptors."+x+")").__name__ == 'function'] if eval("callable(__import__('rdkit').Chem.Descriptors."+y+')')]]      
  dimension=len(raw_value[0])


if __name__ == "__main__":
  
  parse_smi(fname)
  
  #parse_csv(fname)
  

  raw_value=np.array(raw_value)

 
  print('len data = {}, len data[0] = {}'.format(len(raw_value),len(raw_value[0])))
  pca = PCA().fit(preprocessing.scale(raw_value))

  i=0

  for eigenvalue,eigenvector in zip(pca.explained_variance_,pca.components_):
    print('eigenvector = {}, eigenvalue = {}, ratio = {}'.format(eigenvector,eigenvalue,pca.explained_variance_ratio_[i]))
    i+=1
  i=0
  cov_mat=np.cov(preprocessing.scale(raw_value), rowvar=False)
  print('len cov_mat = {}, len cov_mat[0]={}'.format(len(cov_mat),len(cov_mat[0])))
  print('components_ ',pca.components_)
  print('len components_ = {}, len components_[0] = {}'.format(len(pca.components_),len(pca.components_[0])))
  plt.rcParams["figure.figsize"] = (12,6)

  
  cut_off=80 # change this to plot more variance point
  fig, ax = plt.subplots()
  xi = np.arange(1, dimension+1, step=1)[:cut_off]
  y = np.cumsum(pca.explained_variance_ratio_)[:cut_off]

  print(len(xi))
  print(y)
  print(pca.explained_variance_ratio_)
  plt.ylim(0.0,1.1)
  plt.plot(xi, y, marker='o', linestyle='--', color='b')

  plt.xlabel('Number of Components')
  plt.xticks(np.arange(0, cut_off, step=2)) #change from 0-based array index to 1-based human-readable label
  plt.ylabel('Cumulative variance (%)')
  plt.title('The number of components needed to explain variance')

  plt.axhline(y=0.95, color='r', linestyle='-')
  plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

  ax.grid(axis='x')
  plt.show()


  data=preprocessing.scale(raw_value)
  data_out= runPCA(data,int(input('Number of PCA components: ')))



  # 0 = DBSCAN, 1 = Hierarchical
  clst_type = 0
  clustering = do_clustering(clst_type,data_out)
  silhouette_score = clusters_evaluation(clustering,data_out)
  col = 0
  
  for label in set(clustering.labels_):
    print(label)
    points = [data_out[i] for i in range(len(data_out)) if clustering.labels_[i] == label]
    indexes = [i for i in range(len(data_out)) if clustering.labels_[i] == label]
    plt.scatter([el[0] for el in points],[el[1] for el in points],label=label,c=rand_col())
    
    plt.legend()
    print('{} => {}'.format(label,[cmp_name[i] for i in indexes ]))
  print("silhouette_score for clustering is {}".format(silhouette_score))

 