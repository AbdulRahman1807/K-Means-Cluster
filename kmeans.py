import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv('Mall_Customers.csv')
print(df.head())
print(df.info())
print(df.describe())

clusters = {
    0: "Medium Income, Medium Spending",
    1: "High Income, High Spending",
    2: "Low Income, High Spending",
    3: "High Income, Low Spending",
    4: "Low Income, Low Spending"
}

plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'],color="orange")
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Distribution')
plt.show()

x=df[['Annual Income (k$)','Spending Score (1-100)']]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
print(x)

wcss=[]
for i in range(1,11):
    model = KMeans(n_clusters=i,random_state=42)
    model.fit(x_scaled)
    wcss.append(model.inertia_)

plt.plot(range(1,11),wcss,marker='o')
plt.show()

kmeans = KMeans(n_clusters=5,random_state=42)
df['Cluster']=kmeans.fit_predict(x_scaled)
print(df.head())

plt.figure(figsize=(18,10))
for i in range(5):
    plt.scatter(df[df['Cluster']==i]['Annual Income (k$)'],
                df[df['Cluster']==i]['Spending Score (1-100)'],
                label=f"Cluster {i}: {clusters[i]}"
                )
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.legend()
plt.show()

joblib.dump(scaler,"scaler.pkl")
joblib.dump(kmeans,"model.pkl")

