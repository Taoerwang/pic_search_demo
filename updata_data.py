from milvus import Milvus, IndexType, MetricType, Status
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os
from keras import backend as K
import pandas as pd


data_name = 'D:/py_project/untitled1/polls/homework/vae/data.csv'
data = pd.read_csv(data_name)
vectors = data.values
# print(np.shape(data))
vector_ids = [id for id in range(60000)]

milvus = Milvus()
milvus.connect(host='localhost', port='19530')
collection_name = 'mnist'

param = {'collection_name': collection_name, 'dimension': 2, 'index_file_size': 1024, 'metric_type': MetricType.L2}
milvus.create_collection(param)
milvus.insert(collection_name=collection_name, records=vectors, ids=vector_ids)



# search_param = {'nprobe': 16}
# q_records = [[0.3364408, 0.20656677]]
# status, result = milvus.search(collection_name=collection_name, query_records=q_records, top_k=5, params=search_param)
#
# for row in result:
#
#     for item in row:
#         print("id={}, distance={}".format(item.id, item.distance))
#
#
milvus.disconnect()

