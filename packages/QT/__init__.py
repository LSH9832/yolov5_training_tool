import os

def ui2main(ui_name, file_name = 'main.py', model_name = '*_ui2py.py'):
    if model_name == '*_ui2py.py':
        model_name = ui_name.split('.')[0] + '_ui2py.py'

    if os.path.exists(ui_name):
        if os.path.exists(model_name):
            print('由ui直接转化为py格式的文件' + model_name + '已存在')
        else:
            print('开始转化ui文件至:' + model_name)
            # os.system('pyuic5 -o ' + model_name + ' ' + ui_name)
            os.system('python3 -m PyQt5.uic.pyuic -o ' + model_name + ' ' + ui_name)

            while True:
                if os.path.exists(model_name):
                    break

        if os.path.exists(file_name):
            print('用于编写功能的py文件(运行此函数)' + file_name + '已存在')
        else:
            print('开始生成主函数文件至:' + file_name)
            model_text = open(model_name,encoding='utf8').read().split('\n')
            msg = {'model_class':'未识别出', 'button':[], 'action':[], 'combobox':[]}

            for line in model_text:
                if 'class ' in line:
                    msg['model_class'] = line.split(' ')[1].split('(')[0]
                elif 'QtWidgets.QPushButton(' in line:
                    button_name = line.split(' = ')[0]  # .replace(' ','')
                    msg['button'].append(button_name)
                elif '= QtWidgets.QAction(' in line:
                    action_name = line.split(' = ')[0]
                    msg['action'].append(action_name)
                elif 'QtWidgets.QComboBox(' in line:
                    combobox_name = line.split(' = ')[0]
                    msg['combobox'].append(combobox_name)

            buttonactive_test = '\n        # 激活全部按钮、菜单选项、下拉列表用于测试，实际使用时注释掉\n'
            button_text = '\n        # 事件连接函数\n'
            buttonfun_text = '\n\n    # 按钮\n'
            for button in msg['button']:
                buttonactive_test += button + '.setEnabled(True)\n'
                button_text += button +'.clicked.connect(' + button.replace(' ','') + '_ClickFun)\n'
                buttonfun_text += '    def ' + button.replace(' ','').replace('self.', '') + '_ClickFun(self):\n        print("你按了 " + ' + button.replace(' ','') + '.text() + " 这个按钮")\n\n'



            actionactive_test = '\n'
            action_text = '\n'
            actionfun_text = '\n    # 菜单选项\n'
            for action in msg['action']:
                actionactive_test += action + '.setEnabled(True)\n'
                action_text += action + '.triggered.connect(' + action.replace(' ', '') + '_ClickFun)\n'
                actionfun_text += '    def ' + action.replace(' ', '').replace('self.',
                                                                               '') + '_ClickFun(self):\n        print("你按了 " + ' + action.replace(
                    ' ', '') + '.text() + " 这个菜单选项")\n\n'

            comboboxactive_test = '\n'
            combobox_text = '\n'
            comboboxfun_text = '\n    # 下拉列表\n'
            for combobox in msg['combobox']:
                comboboxactive_test += combobox + '.setEnabled(True)\n'
                combobox_text += combobox + '.currentIndexChanged.connect(' + combobox.replace(' ', '') + '_ClickFun)\n'
                comboboxfun_text += '    def ' + combobox.replace(' ', '').replace('self.',
                                                                               '') + '_ClickFun(self):\n        print("你将该下拉列表选项变成了 " + ' + combobox.replace(
                    ' ', '') + '.currentText())\n\n'


            sum_test = buttonactive_test + actionactive_test + comboboxactive_test +\
                       button_text + action_text + combobox_text +\
                       buttonfun_text + actionfun_text + comboboxfun_text

            file_text = open(str(__file__).replace('__init__.py', 'model.txt'), encoding='utf8').read()
            file_text = file_text.replace('MyFunction',
                                          str(os.path.realpath(__file__)).replace('\\', '/').split('/')[-3])
            file_text = file_text.replace('模板类', msg['model_class'])
            file_text = file_text.replace('模板', model_name.split('.')[0])
            file_text = file_text.replace('此处是一堆连接', sum_test)









            open(file_name, 'w+', encoding='utf8').write(file_text)
            print('完成')


    else:
        print('文件' + ui_name + '不存在！程序退出')

# if __name__ == '__main__':
#     CreateWritngFile('main.ui', 'test.py')