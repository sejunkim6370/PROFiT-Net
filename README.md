PRoperty-networking Orbital Field maTrix-convolutional neural Network (PROFiT-Net)

# Package
python==3.8 

tensorflow==2.2.0

numpy==1.18.5

pymatgen==2023.2.22

matminer==0.7.4

# Setup for PROFiT
cp matrix.py cgcnn_atom_feature.json Polarizability.table matminer/featurizers/structure

# Other models
Crystal Graph Convolutional Neural Networks (CGCNN): https://github.com/txie-93/cgcnn [Phys. Rev. Lett. 120, 145301 (2018)]

MatErials Graph Network (MEGNet): https://github.com/materialsvirtuallab/megnet [Chem. Mater. 31, 3564−3572 (2019)]

Orbital Graph Convolutional Neural Network (OGCNN): https://github.com/RishikeshMagar/OGCNN [Phys. Rev. Mater. 4, 093801 (2020)]

# Produce data from a CIF file
python data.py

# Train and validate the model
python main.py

# Predict a target property using the model
python predict.py

# Request pretrained models
Contact: linus16@kaist.ac.kr
