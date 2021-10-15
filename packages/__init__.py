print('\033[1;31m该模块位置：\033[0m\033[1;32m' + str(__file__).replace('\\','/').replace('/__init__.py','') + '\033[0m  \033[1;31m  Loading... \033[0m')

# 将其他格式的图片转换为ico格式的图标文件
def img2ico(img='*.jpg', ico='*.ico', size = (128,128)):
    import os
    from PIL import Image
    if ico == '*.ico':
        ico = img.split('.')[0] + '.ico'
    if img == '*.jpg' and not os.path.exists(img):
        print('From MyFunction.img2ico: 请至少输入图片名称！')
    else:
        icon_sizes = [size]
        im = Image.open(img)
        directory, _ = os.path.split(img)
        out_path = os.path.join(directory, ico)
        im.save(out_path, sizes=icon_sizes)

# 将图片保存为py文件，import后使用GetImg()获得图片
def img2py(picture_name):
    import base64
    """
    将图像文件转换为py文件
    :param picture_name:
    :return:
    """
    open_pic = open("%s" % picture_name, 'rb')
    b64str = base64.b64encode(open_pic.read())
    open_pic.close()
    # 注意这边b64str一定要加上.decode()
    write_data = 'import base64\n' \
                 'from PIL import Image\n' \
                 'from io import BytesIO\n\n' +\
                 'def GetImg():\n' \
                 '    img = "%s"\n' % b64str.decode() + \
                 '    return Image.open(BytesIO(base64.b64decode(img)))\n'
    f = open('%s.py' % picture_name.split('.')[0], 'w+')
    f.write(write_data)
    f.close()

# 两矩形框重合度
XYWH = 1
XYXY = 0
def overlap(loc1:tuple, loc2:tuple, datatype = XYWH):
    x1,y1,w1,h1 = loc1
    x2,y2,w2,h2 = loc2
    left_top = (max(x1,x2),max(y1,y2))
    if datatype:
        right_down = (min(x1+w1,x2+w2),min(y1+h1,y2+h2))
    else:
        right_down = (min(w1, w2), min(h1, h2))

    s_and = 0
    if right_down[0] > left_top[0] and right_down[1] > left_top[1]:
        s_and = (right_down[0] - left_top[0]) * (right_down[1] - left_top[1])
    s_or = w1 * h1 + w2 * h2 - s_and
    return float(s_and) / float(s_or)
