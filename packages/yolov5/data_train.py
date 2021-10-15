import os
import random
import xml.etree.ElementTree as ET
import numpy as np
import argparse
from glob import glob

from .kmeans import kmeans, avg_iou
from .train import main as start_train, device_count


this_dir = str(os.path.realpath(__file__)).replace(str(os.path.realpath(__file__)).replace('\\', '/').split('/')[-1], '')

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def options(weights, cfg, data, epochs, batch_size, address, hyp=None, imgsz = 640, known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=weights, help='initial weights path')
    parser.add_argument('--cfg', type=str, default=cfg, help='model.yaml path')
    parser.add_argument('--data', type=str, default=data, help='dataset.yaml path')

    hyp = this_dir + 'default_hyp.yaml' if hyp is None or not os.path.exists(hyp) else hyp
    parser.add_argument('--hyp', type=str, default=hyp, help='hyperparameters path')
    parser.add_argument('--default_hyp', type=str, default=this_dir + 'default_hyp.yaml', help='hyperparameters path')

    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--batch-size', type=int, default=batch_size, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=imgsz, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default=address + '/runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

class Preparation(object):

    def __init__(self, dir_name = 'my_data', address=this_dir, classes = ('a', 'b')):
        if address.startswith("~"):
            address = "/home/" + os.popen("echo $USERNAME").read().strip("\n") + address[1:]
        if not (address.endswith('/') or address.endswith('\\')):
            address += '/'
        address.replace('\\', '/')
        self.data_address = address
        self.dir_name = dir_name
        self.classes = classes

        self.sets = ['train', 'val', 'test']
        self.son_dirs = ['images', 'Annotations', 'ImageSets', 'ImageSets/Main', 'labels']
        self.yaml_anchors = """  - [data1, data2, data3]  # P3/8
  - [data4, data5, data6]  # P4/16
  - [data7, data8, data9]  # P5/32"""

    def convert_annotation(self, image_id):
        in_file = open(self.data_address + self.dir_name + '/' + self.son_dirs[1] + '/%s.xml' % image_id, encoding='UTF-8')
        out_file = open(self.data_address + self.dir_name + '/' + self.son_dirs[4] + '/%s.txt' % image_id, 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            # difficult = obj.find('difficult').text
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes or int(difficult) == 1:
                continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            b1, b2, b3, b4 = b
            # 标注越界修正
            if b2 > w:
                b2 = w
            if b4 > h:
                b4 = h
            b = (b1, b2, b3, b4)
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    def create_data_dir(self):
        os.mkdir(self.data_address + self.dir_name)
        text = 'Preparation Step\n' \
               '1. Put your images into dir images\n' \
               '2. Put your all relative labels(.xml) into dir Annotations\n' \
               '3. Put label name file(label.txt, one label name for each line) into this dir'
        txt_file = open(self.data_address + self.dir_name + '/READ_ME.txt', 'w')
        txt_file.write(text)
        txt_file.close()
        for son_dir in self.son_dirs:
            dir_to_make = self.data_address + self.dir_name + '/' + son_dir
            if not os.path.exists(dir_to_make):
                os.mkdir()

    def split_data(self, trainval_percent = 1.0, train_percent = 0.9):
        total_xml = os.listdir(self.data_address + self.dir_name + '/' + self.son_dirs[1])
        txtsavepath = self.data_address + self.dir_name + '/' + self.son_dirs[3]
        num = len(total_xml)
        list_index = range(num)
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(list_index, tv)
        train = random.sample(trainval, tr)

        file_trainval = open(txtsavepath + '/trainval.txt', 'w')
        file_test = open(txtsavepath + '/test.txt', 'w')
        file_train = open(txtsavepath + '/train.txt', 'w')
        file_val = open(txtsavepath + '/val.txt', 'w')

        for i in list_index:
            name = total_xml[i][:-4] + '\n'
            if i in trainval:
                file_trainval.write(name)
                if i in train:
                    file_train.write(name)
                else:
                    file_val.write(name)
            else:
                file_test.write(name)

        file_trainval.close()
        file_train.close()
        file_val.close()
        file_test.close()

    def create_voc_label(self):
        for image_set in self.sets:
            image_ids = open(self.data_address + self.dir_name + '/' + self.son_dirs[3] + '/%s.txt' % image_set).read().strip().split()
            list_file = open(self.data_address + self.dir_name + '/' + '%s.txt' % image_set, 'w')
            for image_id in image_ids:
                list_file.write(self.data_address + self.dir_name + '/' + self.son_dirs[0] + '/%s.jpg\n' % image_id)
                self.convert_annotation(image_id)
            list_file.close()

    def create_yaml(self):
        file_name = self.data_address + self.dir_name + '/' + self.dir_name + '.yaml'
        yaml_file = open(file_name, 'w')

        stringclass = "["
        for name in self.classes:
            stringclass += "'" + name + "', "
        stringclass = stringclass[:-2]
        stringclass+="]"

        string = "train: " + self.data_address + self.dir_name + "/" + "train.txt\n" \
                 "val: "  + self.data_address + self.dir_name + '/' + "val.txt\n" \
                 "\n" \
                 "nc: " + str(len(self.classes)) + "\n" \
                 "names: " + stringclass + "\n"

        yaml_file.write(string)
        yaml_file.close()

    def load_data(self, anno_dir, class_names):
        xml_names = os.listdir(anno_dir)
        boxes = []
        for xml_name in xml_names:
            xml_pth = os.path.join(anno_dir, xml_name)
            tree = ET.parse(xml_pth)

            width = float(tree.findtext("./size/width"))
            height = float(tree.findtext("./size/height"))

            for obj in tree.findall("./object"):
                cls_name = obj.findtext("name")
                if cls_name in class_names:
                    xmin = float(obj.findtext("bndbox/xmin")) / width
                    ymin = float(obj.findtext("bndbox/ymin")) / height
                    xmax = float(obj.findtext("bndbox/xmax")) / width
                    ymax = float(obj.findtext("bndbox/ymax")) / height

                    box = [xmax - xmin, ymax - ymin]
                    boxes.append(box)
                else:
                    continue
        return np.array(boxes)

    def clauculate_anchors(self, k=9, loop=1, d=None):
        ANNOTATION_PATH = self.data_address + self.dir_name + '/' + self.son_dirs[1]
        ANCHORS_TXT_PATH = self.data_address + self.dir_name + '/' + "anchors.txt"
        CLUSTERS = k
        CLASS_NAMES = self.classes

        anchors_txt = open(ANCHORS_TXT_PATH, "w")

        train_boxes = self.load_data(ANNOTATION_PATH, CLASS_NAMES)
        count = 1
        best_accuracy = 0
        best_anchors = []
        best_ratios = []

        for i in range(loop):       # 可以修改，不要太大，否则时间很长
            anchors_tmp = []
            clusters = kmeans(train_boxes, k=CLUSTERS, d=d)
            idx = clusters[:, 0].argsort()
            clusters = clusters[idx]
            # print(clusters)

            for j in range(CLUSTERS):
                anchor = [round(clusters[j][0] * 640, 2), round(clusters[j][1] * 640, 2)]
                anchors_tmp.append(anchor)
                # print(f"Anchors:{anchor}")

            temp_accuracy = avg_iou(train_boxes, clusters) * 100
            # print("Train_Accuracy:{:.2f}%".format(temp_accuracy))

            ratios = np.around(clusters[:, 0] / clusters[:, 1], decimals=2).tolist()
            ratios.sort()
            # print("Ratios:{}".format(ratios))
            # print(20 * "*" + " {} ".format(count) + 20 * "*")

            count += 1

            if temp_accuracy > best_accuracy:
                best_accuracy = temp_accuracy
                best_anchors = anchors_tmp
                best_ratios = ratios

        anchors_txt.write("Best Accuracy = " + str(round(best_accuracy, 2)) + '%' + "\r\n")
        anchors_txt.write("Best Anchors = " + str(best_anchors) + "\r\n")
        anchors_txt.write("Best Ratios = " + str(best_ratios))
        anchors_txt.close()
        i = 0

        for this_anchor in best_anchors:
            i += 1
            self.yaml_anchors = self.yaml_anchors.replace("data" + str(i), str(round(this_anchor[0])) + ',' + str(round(this_anchor[1])))

    def create_cfg(self, net_type = 'all'):
        all_type = ["s", "m", "l", "x"]
        if net_type == "all":
            for this_type in all_type:
                this_text = open(this_dir + 'train_net_model/yolov5' + this_type + '.yaml').read()
                this_text = this_text.replace("anchorsdata", self.yaml_anchors).replace("ncdata", str(len(self.classes)))
                yaml_file = open(self.data_address + self.dir_name + '/yolov5' + this_type + '.yaml', 'w')
                yaml_file.write(this_text)
                yaml_file.close()
        elif net_type in all_type:
            this_type = net_type
            this_text = open(this_dir + 'train_net_model/yolov5' + this_type + '.yaml').read()
            this_text = this_text.replace("anchorsdata", self.yaml_anchors).replace("ncdata", str(len(self.classes)))
            yaml_file = open(self.data_address + self.dir_name + '/yolov5' + this_type + '.yaml', 'w')
            yaml_file.write(this_text)
            yaml_file.close()

    def get_train_options(self, net_type, epochs = 300, batch_size = 16, imgsz = 640, othermodel = None):
        all_type = ["s", "m", "l", "x"]
        if net_type in all_type:
            print("\nweights = ", this_dir + 'models/pt/yolov5' + net_type + '.pt' if othermodel is None else othermodel,
                  "\ncfg =", self.data_address + self.dir_name + '/yolov5'  + net_type + '.yaml',
                  "\ndata = ", self.data_address + self.dir_name + '/'  + self.dir_name  + '.yaml',
                  "\nepochs = ", epochs,
                  "\nbatch_size = ", batch_size,
                  "\nhyp = ", self.data_address + self.dir_name + '/hyp.yaml',
                  "\nimgsz = ", imgsz)
            return options(weights = this_dir + 'models/pt/yolov5' + net_type + '.pt' if othermodel is None else othermodel,
                           cfg = self.data_address + self.dir_name + '/yolov5'  + net_type + '.yaml',
                           data = self.data_address + self.dir_name + '/'  + self.dir_name  + '.yaml',
                           epochs = epochs,
                           batch_size = batch_size,
                           address = self.data_address + self.dir_name,
                           hyp = self.data_address + self.dir_name + '/hyp.yaml',
                           imgsz = imgsz)


    def check_data(self):
        ret = True
        msg = 'ok'
        img_list = sorted(glob(self.data_address + self.dir_name + '/' + self.son_dirs[0] + '/*'))
        label_list = sorted(glob(self.data_address + self.dir_name + '/' + self.son_dirs[1] + '/*'))
        print(self.data_address + self.dir_name + '/' + self.son_dirs[0] + '/*.jpg')
        if len(img_list) == len(label_list) == 0:
            ret = False
            msg = 'Neither images nor labels put into correct path'
        elif not len(img_list)==len(label_list):
            ret = False
            msg = 'Number of images and labels not equal'
        else:
            # print(img_list, label_list)
            for i in range(len(img_list)):
                this_img = img_list[i].replace("\\","/").split("/")[-1].split(".")[0]
                this_label = label_list[i].replace("\\","/").split("/")[-1].split(".")[0]
                if not this_img == this_label:
                    ret = False
                    msg = 'Name of images and labels not same'
                    break
        return ret, msg


if __name__ == '__main__':
    my_pre = Preparation(address='/home/shl/code/project/yolov5train/yolov5/', classes=('person', 'gun', 'oil drum', 'caisson'))
    try:
        my_pre.create_data_dir()
    except:
        pass
    # my_pre.split_data()           # 将数据集分为训练集和测试集
    # my_pre.create_voc_label()     # 将xml文件转化为yolo能够识别的格式的数据并保存为txt文件于label文件夹中
    # my_pre.create_yaml()          # 在主目录中创建含有数据集信息的yaml文件
    print(os.path.realpath(__file__))
    # my_pre.clauculate_anchors()   # 计算最佳anchors并保存在主目录anchors.txt中
