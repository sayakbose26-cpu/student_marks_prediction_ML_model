import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
df = pd.read_csv("dataset/student-mat.csv",sep=";")
sns.histplot(data=df, x="G3",bins=20,color="blue", kde=True)
plt.show()
