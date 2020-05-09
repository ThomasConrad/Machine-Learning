import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[10,4],[5,12]]        
df_cm = pd.DataFrame(array, range(2),
                  range(2))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16},cmap='binary',, columns=["True","False"])# font size




plt.show()