## Script for downloading data

# bbox Annotations
wget -P data http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip data/v2_Annotations_Train_mscoco.zip -d data
rm data/v2_Annotations_Train_mscoco.zip

#images train and val
wget -P images http://images.cocodataset.org/zips/train2014.zip
unzip iamges/train2014.zip -d images
rm iamges/train2014.zip
wget -P images http://images.cocodataset.org/zips/val2014.zip
unzip iamges/val2014.zip -d images
rm iamges/val2014.zip
