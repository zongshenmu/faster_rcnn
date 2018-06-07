#encoding=utf-8

import json
import pickle

path = './data/instances_{}2014.json'

def get_ctgs():
    with open(path.format('val'), 'r') as file:
        instances = json.load(file)
    ctgs = instances['categories']
    categories = {}
    for ctg in ctgs:
        categories[ctg['id']] = ctg['name']
    with open('./data/categories.pkl', 'wb') as wf:
        pickle.dump(categories, wf)

def collect_data():
    names = ['train', 'val']
    print('process start')
    #获取classname
    with open('./data/categories.pkl', 'rb') as rf:
        categories = pickle.load(rf)
    #print(categories)
    #获取标记的bbox
    for name in names:
        with open(path.format(name), 'r') as file:
            instances = json.load(file)
            #处理bbox
            annotations = instances['annotations']
            datas = []
            for annotation in annotations:
                per_data = {}
                per_data['bbox'] = annotation['bbox']
                #print(categories[annotation['category_id']])
                per_data['ctg_id'] = categories[annotation['category_id']]
                per_data['img_id'] = annotation['image_id']
                datas.append(per_data)
                del per_data
            with open('./data/{}_data.pkl'.format(name), 'wb') as wf:
                pickle.dump(datas, wf)
            del datas
            #处理图片长宽
            raw_images=instances['images']
            images={}
            for image in raw_images:
                w=image['width']
                h=image['height']
                images[image['id']]={
                    'w':w,
                    'h':h
                }
            with open('./data/{}_images.pkl'.format(name), 'wb') as wf:
                pickle.dump(images, wf)
    print('process over')

if __name__ == '__main__':
    collect_data()