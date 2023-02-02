from scipy.stats import ranksums
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np



class MidpointNormalize(mpl.colors.Normalize):
  def __init__(self, vmin, vmax, midpoint=0, clip=False):
    self.midpoint = midpoint
    mpl.colors.Normalize.__init__(self, vmin, vmax, clip)


  def __call__(self, value, clip=None):
    normalized_min = max(0,
        0.5 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
    
    normalized_max = min(1,
        0.5 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
    
    normalized_mid = 0.5
    
    x = [self.vmin, self.midpoint, self.vmax]
    y = [normalized_min, normalized_mid, normalized_max]
    return np.ma.masked_array(np.interp(value, x, y))



class AnalysisIntern:
  def __init__(self, cluster_df_list, var_type="cont"):
    self.var_type = var_type
    self.cluster_df_list  = cluster_df_list


  def p_val(self, clustera, clusterb, feature):
    if self.var_type == 'cont':
      return stats.ttest_ind(
        np.array(clustera[feature]) , np.array(clusterb[feature])).pvalue
    else:
      return ranksums(
        np.array(clustera[feature]) , np.array(clusterb[feature])).pvalue


  def feature_p_val(self, feature):
    return np.array([
      np.array([self.p_val(ci, cj, feature) for cj in self.cluster_df_list])
      for ci in self.cluster_df_list
      ])


  def p_map(self, feature):
    heatmap, ax = plt.subplots(figsize=(8, 8), dpi=600)
    norm = MidpointNormalize(vmin=0, vmax=1, midpoint=0.5)

    p_values = self.feature_p_val(feature)

    im = ax.imshow(p_values , cmap='coolwarm' , norm=norm)
    ax.set_xticklabels(['','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    ax.set_yticklabels(['','C1','C2','C3','C4','C5','C6','C7','C8','C9'])

    num_clusters = len(self.cluster_df_list)
    for y in range(num_clusters):
      for x in range(num_clusters):
        plt.text(x , y
          , '%.2f' % p_values[y, x]
          , horizontalalignment='center'
          , verticalalignment='center'
          , fontsize=8
          )

    cbar = heatmap.colorbar(im)
    cbar.ax.set_ylabel('p-value')
    plt.title(feature.upper(), fontsize=16)
    print('\n')
    plt.show()




class Analysis:
  def __init__(self
               , cont_features, ord_features, nom_features
               , clusters_dbscan_FDC, values_dbscan_FDC):
    self.cont_features = cont_features
    self.ord_features = ord_features
    self.nom_features = nom_features
    self.clusters_dbscan_FDC = clusters_dbscan_FDC
    self.values_dbscan_FDC = values_dbscan_FDC

    self.cluster_df_list = []


  def run(self, data):
    data['Clusters'] = np.array(self.clusters_dbscan_FDC)

    cluster_df_list=[]
    for cluster in self.values_dbscan_FDC:
      cluster_df = data.loc[data['Clusters'] == cluster].drop(columns=['Clusters'])
      cluster_df.columns = list(data.columns)[:-1]
      cluster_df_list.append(cluster_df)

    self.cluster_df_list = cluster_df_list[1:]

    a = AnalysisIntern(self.cluster_df_list, "cont")
    for feature in self.cont_features:
      a.p_map(feature)

    a = AnalysisIntern(self.cluster_df_list, "ord")
    for feature in self.ord_features:
      a.p_map(feature)

    for feature in self.nom_features:
      a.p_map(feature)
