import pandas as pd
import numpy as np
import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph,StellarDiGraph
from stellargraph import datasets
from sklearn import model_selection
from IPython.display import display, HTML
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf
from skimage import io
import warnings
warnings.filterwarnings("ignore")

#######################################################
# Load data and process

main_path = ""  # main dir of data
label_paths = ["", ""] # label of data, including benign and malignant
direct_paths = ["","","",""] # the sub class of data
names, thyroids, nodules, labels = [], [], [], []
for label_path in label_paths:
    H_V_path = ""
    for name in os.listdir(main_path+label_path+H_V_path+"nodule"):
        thyroid = io.imread(main_path+label_path+H_V_path+"thyroid/"+name)
        thyroid = feature.canny(thyroid,sigma=0.5)*255
        thyroids.append(thyroid)
            
        nodule = io.imread(main_path+label_path+H_V_path+"nodule/"+name)
        nodule = feature.canny(nodule,sigma=0.5)*255
        nodules.append(nodule)
                
        names.append(name)
        if label_path == "":
            labels.append(1)
        elif label_path == "":
            labels.append(0)
        else:
            print("Error in: "+label_path)

#######################################################
# get point of nodule and thyroid
nodule_hulls,thyroid_points=[],[]
labels_hull,names_hull,nodules_hull,thyroids_hull=[],[],[],[]
len_benign_nodule, len_malignant_nodule=0, 0
for i in range(len(nodules)):
    nodule=nodules[i]
    nodule_index=np.where(nodule==255)
    nodule_point=np.zeros((len(nodule_index[0]),2))
    nodule_point[:,0]=nodule_index[0]
    nodule_point[:,1]=nodule_index[1]
    
    thyroid=thyroids[i]
    thyroid_index=np.where(thyroid==255)
    thyroid_point=np.zeros((len(thyroid_index[0]),2))
    thyroid_point[:,0]=thyroid_index[0]
    thyroid_point[:,1]=thyroid_index[1]
    try:
        nodule_hull=ConvexHull(nodule_point)
        if labels[i]==0:
            len_benign_nodule+=1
        elif labels[i]==1:
            len_malignant_nodule+=1
        else:
            print("Error in label: "+str(labels[i]))
            
        nodule_hulls.append(nodule_hull)   
        thyroid_points.append(thyroid_point)
        
        labels_hull.append(labels[i])
        names_hull.append(names[i])
        nodules_hull.append(nodules[i])
        thyroids_hull.append(thyroids[i])
    except:
        print("Error in file: "+names[i])
        print(i)
        
#### get hull point of nodule
nodule_indexs=[]
nodule_convex_indexs=[]
nodule_x_locs=[]
nodule_y_locs=[]
for i in range(len_benign_nodule+len_malignant_nodule):
    nodule_index=[]
    nodule_convex_index=[]
    nodule_x_loc=[]
    nodule_y_loc=[]
    nodule_index.append(i)
    for j in range(len(nodule_hulls[i].vertices)):
        nodule_convex_index.append(j)
        nodule_x_loc.append(nodule_hulls[i].points[nodule_hulls[i].vertices[j]][0])
        nodule_y_loc.append(nodule_hulls[i].points[nodule_hulls[i].vertices[j]][1])
    nodule_indexs.append(nodule_index)
    nodule_convex_indexs.append(nodule_convex_index)
    nodule_x_locs.append(nodule_x_loc)
    nodule_y_locs.append(nodule_y_loc)

#### get all point of thyroid
thyroid_indexs=[]
thyroid_point_indexs=[]
thyroid_x_locs=[]
thyroid_y_locs=[]
sample_number=300
for i in range(len_benign_nodule+len_malignant_nodule):
    thyroid_index=[]
    thyroid_point_index=[]
    thyroid_x_loc=[]
    thyroid_y_loc=[]
    thyroid_len=len(thyroid_points[i])
    ratio=int(thyroid_len/sample_number)
    thyroid_index.append(i)
    for j in range(sample_number):
        thyroid_point_index.append(j)
        thyroid_x_loc.append(thyroid_points[i][ratio*j][0])
        thyroid_y_loc.append(thyroid_points[i][ratio*j][1])
    thyroid_indexs.append(thyroid_index)
    thyroid_point_indexs.append(thyroid_point_index)
    thyroid_x_locs.append(thyroid_x_loc)
    thyroid_y_locs.append(thyroid_y_loc)
    
#### generate the dictonary
nodule_dict={"nodule_indexs": nodule_indexs,
           "nodule_convex_indexs": nodule_convex_indexs,
           "nodule_x_locs": nodule_x_locs,
           "nodule_y_locs": nodule_y_locs}
nodule_df=DataFrame(nodule_dict)

thyroid_dict={"thyroid_indexs":thyroid_indexs,
           "thyroid_point_indexs": thyroid_point_indexs,
           "thyroid_x_locs": thyroid_x_locs,
           "thyroid_y_locs": thyroid_y_locs}
thyroid_df=DataFrame(thyroid_dict)

#######################################################
# Generate the graph from point, and fit the DGCNN model
nodule_num # the number of your data
graphs=[]
graph_labels=[]
for i in range(nodule_num):
    # edge_data
    source_list=[]
    target_list=[]
    dist_list=[]
    for p in range(len(nodule_df["nodule_convex_indexs"][i])):
        for q in range(len(thyroid_df["thyroid_point_indexs"][i])):
            source_list.append("convex_"+str(nodule_df["nodule_convex_indexs"][i][p]))
            target_list.append("point_"+str(thyroid_df["thyroid_point_indexs"][i][q]))
            dist=(nodule_df["nodule_x_locs"][i][p]-thyroid_df["thyroid_x_locs"][i][q])**2+\
            (nodule_df["nodule_y_locs"][i][p]-thyroid_df["thyroid_y_locs"][i][q])**2
            dist_list.append(dist)
    edge_data=pd.DataFrame(
        {
         "source":source_list,
         "target":target_list,
         "dist":dist_list
        }
    )
    
    # nodule_data
    nodule_index=nodule_df["nodule_indexs"][i]
    single_nodule_list=[]
    single_x_loc_list=[]
    single_y_loc_list=[]
    for j in range(len(nodule_df["nodule_convex_indexs"][i])):
        single_nodule_list.append("convex_"+str(nodule_df["nodule_convex_indexs"][i][j]))
        single_x_loc_list.append(nodule_df["nodule_x_locs"][i][j])
        single_y_loc_list.append(nodule_df["nodule_y_locs"][i][j])
    for j in range(len(thyroid_df["thyroid_point_indexs"][i])):
        single_nodule_list.append("point_"+str(thyroid_df["thyroid_point_indexs"][i][j]))
        single_x_loc_list.append(thyroid_df["thyroid_x_locs"][i][j])
        single_y_loc_list.append(thyroid_df["thyroid_y_locs"][i][j])
    node_data=pd.DataFrame(
        {
         "name":single_nodule_list,
         "x_loc":single_x_loc_list,
         "y_loc":single_y_loc_list
        }
    )
    node_data.index=node_data.iloc[:,0]
    node_data.drop(node_data.columns[0],axis=1,inplace=True)
    square=StellarDiGraph({"corner":node_data},{"line":edge_data})
    graphs.append(square)
    graph_labels.append(name_df["labels_hull"][i])
    print("Finish: {0:d}".format(i+1),end='\r')
    
graph_labels=pd.get_dummies(graph_labels,drop_first=True)
generator = PaddedGraphGenerator(graphs=graphs)
k = 35  # the number of rows for the output tensor
layer_sizes = [32, 32, 32, 1]

dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=128, activation="relu")(x_out)
x_out = Dropout(rate=0.5)(x_out)

predictions = Dense(units=1, activation="sigmoid")(x_out)

epochs = 200
history = model.fit(
    train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,callbacks=callbacks_list
)