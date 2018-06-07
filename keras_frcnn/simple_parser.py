import pickle

#处理的bbox类中不包含background

def get_data(name):
    with open('./data/categories.pkl', 'rb') as rf:
        class_mapping = pickle.load(rf)
        class_mapping = dict(zip(class_mapping.values(),range(len(class_mapping))))
        class_mapping['bg'] =len(class_mapping)+1
    with open('./data/processed_{}_data.pkl'.format(name), 'rb') as tf:
        imgs=pickle.load(tf)
    return imgs, class_mapping