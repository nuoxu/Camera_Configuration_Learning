import xml.etree.ElementTree as ET
import os
import json
import pdb
import pickle
import numpy as np


class xml2json():
    
    def __init__(self):
        self.coco = dict()
        self.coco['images'] = []
        self.coco['type'] = 'instances'
        self.coco['annotations'] = []
        self.coco['categories'] = []

        # self.label_dict = ['F-16', 'J-10', 'Su-33UB', 'E-2c', 
        #                    'YF-22', 'F-14', 'J-5', 'FA-18E', 
        #                    'Boeing787', 'Boeing777', 'Boeing747', 'L-1011']
        self.label_dict = ['Sports-Car', 'Hatchback', 'Pickup', 'SUV', 'Box-Truck']

        self.category_set = dict()
        self.image_set = set()

        self.category_item_id = 0
        self.annotation_id = 0
        self.image_id = 000000

        for lab in self.label_dict:
            self.addCatItem(lab)
    
    def reset(self):

        self.coco = dict()
        self.coco['images'] = []
        self.coco['type'] = 'instances'
        self.coco['annotations'] = []
        self.coco['categories'] = []

        # self.label_dict = ['F-16', 'J-10', 'Su-33UB', 'E-2c', 
        #                    'YF-22', 'F-14', 'J-5', 'FA-18E', 
        #                    'Boeing787', 'Boeing777', 'Boeing747', 'L-1011']
        self.label_dict = ['Sports-Car', 'Hatchback', 'Pickup', 'SUV', 'Box-Truck']

        self.category_set = dict()
        self.image_set = set()

        self.category_item_id = 0
        self.annotation_id = 0
        self.image_id = 000000

        for lab in self.label_dict:
            self.addCatItem(lab)
        return None

    def addCatItem(self, name):
        category_item = dict()
        category_item['supercategory'] = 'none'
        self.category_item_id += 1
        category_item['id'] = self.category_item_id
        category_item['name'] = name
        self.coco['categories'].append(category_item)
        self.category_set[name] = self.category_item_id
        return self.category_item_id
    
    def addImgItem(self, file_name, size):
        if file_name is None:
            raise Exception('Could not find filename tag in xml file.')
        if size['width'] is None:
            raise Exception('Could not find width tag in xml file.')
        if size['height'] is None:
            raise Exception('Could not find height tag in xml file.')
        self.image_id += 1
        image_item = dict()
        image_item['id'] = self.image_id
        # image_item['id'] = int(file_name)
        image_item['file_name'] = file_name + '.jpg'
        image_item['width'] = size['width']
        image_item['height'] = size['height']
        self.coco['images'].append(image_item)
        self.image_set.add(file_name)
        # return int(file_name)
        return self.image_id
    
    def addAnnoItem(self, object_name, image_id, category_id, bbox):
        annotation_item = dict()
        annotation_item['segmentation'] = []
        seg = []
        #bbox[] is x,y,w,h
        #left_top
        seg.append(bbox[0])
        seg.append(bbox[1])
        #left_bottom
        seg.append(bbox[0])
        seg.append(bbox[1] + bbox[3])
        #right_bottom
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1] + bbox[3])
        #right_top
        seg.append(bbox[0] + bbox[2])
        seg.append(bbox[1])
    
        annotation_item['segmentation'].append(seg)
    
        annotation_item['area'] = bbox[2] * bbox[3]
        annotation_item['iscrowd'] = 0
        annotation_item['ignore'] = 0
        annotation_item['image_id'] = image_id
        annotation_item['bbox'] = bbox
        annotation_item['category_id'] = category_id
        self.annotation_id += 1
        annotation_item['id'] = self.annotation_id
        self.coco['annotations'].append(annotation_item)
    
    def parseXmlFiles(self, xml_path, search_dict, step): 
        print(len(os.listdir(xml_path)))
        for f in os.listdir(xml_path):
            if not f.endswith('.xml'):
                continue

            # print(f)
            sf = search_dict[f.split('.')[0]]
    
            if len(sf) < step:
                if len(sf) > 0:
                    f = sf[np.int(len(sf)-1)] + '.xml'
            else:
                if step > 0:
                    f = sf[np.int(step-1)] + '.xml'

            # print(f)
            # print('********')

            bndbox = dict()
            size = dict()
            current_image_id = None
            current_category_id = None
            file_name = None
            size['width'] = None
            size['height'] = None
            size['depth'] = None
    
            xml_file = os.path.join(xml_path, f)
            # print(xml_file)
    
            tree = ET.parse(xml_file)
            root = tree.getroot()
            if root.tag != 'annotation':
                raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))
    
            #elem is <folder>, <filename>, <size>, <object>
            for elem in root:
                current_parent = elem.tag
                current_sub = None
                object_name = None
                
                if elem.tag == 'folder':
                    continue
                
                if elem.tag == 'filename':
                    file_name = elem.text
                    if file_name in self.category_set:
                        raise Exception('file_name duplicated')
                    
                #add img item only after parse <size> tag
                elif current_image_id is None and file_name is not None and size['width'] is not None:
                    if file_name not in self.image_set:
                        file_name = file_name.split('.')[0]
                        current_image_id = self.addImgItem(file_name, size)
                        # print('add image with {} and {}'.format(file_name, size))
                    else:
                        raise Exception('duplicated image: {}'.format(file_name)) 
                #subelem is <width>, <height>, <depth>, <name>, <bndbox>
                for subelem in elem:
                    bndbox ['xmin'] = None
                    bndbox ['xmax'] = None
                    bndbox ['ymin'] = None
                    bndbox ['ymax'] = None
                    
                    current_sub = subelem.tag
                    if current_parent == 'object' and subelem.tag == 'name':
                        object_name = subelem.text
                        if object_name not in self.category_set:
                            current_category_id = self.addCatItem(object_name)
                        else:
                            current_category_id = self.category_set[object_name]
    
                    elif current_parent == 'size':
                        if size[subelem.tag] is not None:
                            raise Exception('xml structure broken at size tag.')
                        size[subelem.tag] = int(subelem.text)
    
                    #option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                    for option in subelem:
                        if current_sub == 'bndbox':
                            if bndbox[option.tag] is not None:
                                raise Exception('xml structure corrupted at bndbox tag.')
                            bndbox[option.tag] = int(option.text)
    
                    #only after parse the <object> tag
                    if bndbox['xmin'] is not None:
                        if object_name is None:
                            raise Exception('xml structure broken at bndbox tag')
                        if current_image_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        if current_category_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        bbox = []
                        #x
                        bbox.append(bndbox['xmin'])
                        #y
                        bbox.append(bndbox['ymin'])
                        #w
                        bbox.append(bndbox['xmax'] - bndbox['xmin'])
                        #h
                        bbox.append(bndbox['ymax'] - bndbox['ymin'])
                        # print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
                        self.addAnnoItem(object_name, current_image_id, current_category_id, bbox )

if __name__ == '__main__':

    xml_path = 'datasets/car/label_test/'
    json_file = 'datasets/car/annotations/instances_val2014_step%d.json'
    with open("ddqn_car_search.pickle", "rb") as fp:
        search_dict = pickle.load(fp)

    xml_to_json = xml2json()
    for i in range(10):
        xml_to_json.parseXmlFiles(xml_path, search_dict, i)
        json.dump(xml_to_json.coco, open(json_file%i, 'w'))
        xml_to_json.reset()