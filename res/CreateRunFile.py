#! /usr/bin/python3
import os
os.system('sudo chmod 777 %s' % str(__file__))
torch_command = {'cpu':'pip3 install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html',
                 '9.2':'pip3 install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html',
                 '10.1':'pip3 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html',
                 '10.2':'pip3 install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0',
                 '11.1':'pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html'}

while True:
    c = input('Have you already install pytorch and its relatives?(y/n)\n>>>')
    if c == 'y' or c == 'Y':
        c= True
        break
    elif c == 'n' or c == 'N':
        c = False
        break
    else:
        print('wrong input!')
# install pytorch
if not c:        
    cudamsg = os.popen('nvcc -V').read()
    if 'release' in cudamsg:
        version = cudamsg.split('release ')[1].split(',')[0]
        print('cuda version:', version)
        os.system(torch_command[version])
    else:
        print('no cuda enviroment, use cpu version')
        os.system(torch_command['cpu'])


this_dir = str(os.path.realpath('./setup.sh')).replace(os.path.realpath('../setup.sh').replace('\\', '/').split('/')[-1],'')
print(this_dir, os.path.realpath('../setup.sh').replace('\\', '/').split('/')[-1])
home_addr = '/home/%s/' % os.popen('echo $USER').read().strip('\n')
desktop_addr = home_addr + 'Desktop' if os.path.exists(home_addr + 'Desktop') else home_addr + '桌面' if os.path.exists(home_addr + '桌面') else '.'


tempFileName = '%s.sh' % os.path.realpath(str(__file__)).replace('\\', '/').split('/')[-3]
tempDeskName = '%s.desktop' % os.path.realpath(str(__file__)).replace('\\', '/').split('/')[-3]
run_file_name = desktop_addr + '/' + tempFileName
deskFileName = home_addr + '.local/share/applications/%s' % tempDeskName
iconName = this_dir + 'icons/icon.png'
tempFileName = 'res/' + tempFileName


while True:
    path = input('if you use a conda env or virtual env, please enter your env path(example:/home/user/anaconda3/envs/myenv) else please press enter without any input\n>>>')
    
    if len(path):
        if path.startswith('~'):
            path = home_addr + path[1:]
        if not path.endswith('/'):
            path += '/'
        if not os.path.exists('%sbin/python3' % path):
            print("interpreter  %sbin/python3 dosen't exitst!" % path)
        else:
            path += 'bin/'
            break
    else:
        break

bash = "cd %s\n%spython3 YOLOv5TrainGuide" % (this_dir, path)

desktop = """[Desktop Entry]
Comment=Yolov5 Train Guide Tool is a opensource application that make training yolov5 model much more easier.
Name=YOLOv5 Train Guide Tool
StartupNotify=false
Exec=sh %s
Terminal=false
Type=Application
Categories=Development;IDE;
Icon=%s""" % (this_dir + tempFileName, iconName)

f1 = open(tempFileName, 'w')
f1.write(bash)
f1.close()
f2 = open(tempDeskName, 'w')
f2.write(desktop)
f2.close()


os.system('sudo cp %s %s' % (tempDeskName, deskFileName))
os.system('sudo chmod 777 %s' % tempFileName)
os.system('sudo chmod 777 %s' % deskFileName)
os.remove(tempDeskName)

























