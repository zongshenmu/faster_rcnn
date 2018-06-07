#first step
run: download.sh
download annotations and images

#second step
you can read keras_frcnn/config.py and revise some parameters wanted

#thrid step
run: python3.6 train_frcnn.py

#final step
if you want directly to run the model, you should firstly download the model weights from
https://drive.google.com/open?id=13wmU30AddIQyvpO0Ln0zmHBgYxTLMywJ,
and then choose you own images to put in the directory of images, at last python3.6 test.py.
