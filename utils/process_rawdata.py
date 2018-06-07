import pickle
import sys
import os

# mscoco数据集中不包含background
def process_data():
    print('Parsing annotation files')
    names = ['train', 'val']
    for name in names:
        with open('./data/{}_images.pkl'.format(name), 'rb') as imgfile:
            images=pickle.load(imgfile)
        with open('./data/{}_data.pkl'.format(name), 'rb') as file:
            datas=pickle.load(file)
            all_imgs={}
            new_data = []
            for data in datas:
                filename=data['img_id']
                x1,y1,w,h=data['bbox']
                x2=x1+w
                y2=y1+h
                class_name=data['ctg_id']
                if filename not in all_imgs:
                    all_imgs[filename] = {}
                    all_imgs[filename]['filepath'] = filename
                    all_imgs[filename]['width'] = images[filename]['w']
                    all_imgs[filename]['height'] = images[filename]['h']
                    all_imgs[filename]['bboxes'] = []

                all_imgs[filename]['bboxes'].append(
                    {'class': class_name,
                     'x1': int(x1),
                     'x2': int(x2),
                     'y1': int(y1),
                     'y2': int(y2)
                     }
                )
            for key in all_imgs:
                new_data.append(all_imgs[key])
            with open('./data/processed_{}_data.pkl'.format(name),'wb') as newfile:
                pickle.dump(new_data,newfile)
    print('Over')

if __name__ == '__main__':
    process_data()