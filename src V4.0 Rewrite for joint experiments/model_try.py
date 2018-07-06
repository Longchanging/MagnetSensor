import pickle
import pandas as pd
import numpy as np

# obj0, obj1, obj2 are created here...
obj0 = {'dfs':[1,2,32],'dsgsd':134}
obj1 = [1,23,2,353,45]
obj2 = np.array([1,2,32])
# Saving the objects:
with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([obj0, obj1, obj2], f)

# Getting back the objects:
with open('objs.pkl','rb') as f:  # Python 3: open(..., 'rb')
    obj0, obj1, obj2 = pickle.load(f)
    print(obj0, obj1, obj2)
    print(type(obj0), type(obj1), type(obj2))

