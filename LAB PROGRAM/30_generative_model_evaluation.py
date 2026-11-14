# CODE:
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
rng=np.random.default_rng(0); real=rng.normal(0,1,(500,2))
gen_a=rng.normal(0,.9,(500,2)); gen_b=rng.normal(.3,1.2,(500,2))
rows=[]
for name,samples in [("Generator A",gen_a),("Generator B",gen_b)]:
 rows.append({"model":name,"mean_distance":float(np.linalg.norm(real.mean(0)-samples.mean(0))),"MSE_to_reference":float(mean_squared_error(real,samples))})
df=pd.DataFrame(rows); print(df)
plt.scatter(real[:,0],real[:,1],s=8,label="Reference"); plt.scatter(gen_a[:,0],gen_a[:,1],s=8,label="Generator A"); plt.scatter(gen_b[:,0],gen_b[:,1],s=8,label="Generator B")
plt.legend(); plt.title("Generated distribution comparison"); plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.savefig("30_generative_model_evaluation.png"); plt.show()

# INPUT:
# Input: synthetic reference distribution and two synthetic generator outputs.
#

# OUTPUT:
# Output: metric dataframe and comparative scatter plot; metrics are illustrative.
