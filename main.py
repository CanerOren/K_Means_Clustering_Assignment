from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


k=int(input("Küme Sayısı:"))

sample=np.genfromtxt("Final-data.csv",delimiter=",",skip_header=1,dtype=float)
sample_norm=(sample - sample.min())/ (sample.max() - sample.min())
km=KMeans(n_clusters=k)
km.fit(sample_norm)
labels=km.predict(sample_norm)
labels_list=labels.tolist()
labels_list_string=[str(x) for x in labels_list]

with open('sonuc.txt', 'w') as f:
    f.write("--------------------------------\n")
    f.write("k= ")
    f.write(str(k))
    f.write(" için\n")
    for i in range(len(labels)):
        f.write("Kayıt:\t")
        f.write(str(i+1))
        f.write("\t \t")
        f.write("Küme:\t")
        #f.write(labels_list_string[i])
        f.write(str(labels[i]))
        f.write("\n")





centroids=km.cluster_centers_
print("*******************")
for i in range(len(labels)):
    print("Kayıt ",i+1,'    ','Küme',labels[i])




columns={"a1":0,"a2":1,"a3":2,"a4":3,"a5":4,"a6":5,"a7":6,"a8":7}
colors=mcolors.BASE_COLORS
for i in range(k):
    points=np.array([sample_norm[j] for j in range (len(sample_norm)) if labels[j]==i])
    plt.scatter(points[:,columns["a1"]],points[:,columns["a2"]],c=list(colors)[i],alpha=0.5,label=f"Cluster{i+1}")


for i in range (k):
    plt.scatter(centroids[i,columns["a1"]],centroids[i,columns["a2"]],c=list(colors)[i],marker='*',label=f"Cluster{i+1} Centroid",s=150)

plt.legend()
plt.show()
