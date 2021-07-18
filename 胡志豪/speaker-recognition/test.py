

def prehelp():
    # 声明一个空字典，来保存文本文件数据
    dict_temp = {}

    # 打开文本文件
    file = open('./train.txt', 'r')

    # 遍历文本文件的每一行，strip可以移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    for line in file.readlines():
        line = line.strip()
        k = line.split(' ')[0]
        v = line.split(' ')[1]
        dict_temp[k] = v

    # 依旧是关闭文件
    file.close()

    #  可以打印出来瞅瞅
    print(dict_temp)
    print(len(dict_temp))
    return  dict_temp;


def reshelp(dict_temp):
    # 先创建并打开一个文本文件
    file = open('./eval_result.txt', 'w')

    # 遍历字典的元素，将每项元素的key和value分拆组成字符串，注意添加分隔符和换行符
    for k, v in sorted(dict_temp.items()) :
        file.write(str(k) + ' ' + str(v) + '\n')

    # 注意关闭文件
    file.close()



    # 字典输出的项是无序的，如果想按照字典的key排序输出的话，可以按照下面的方式实现
   #  for k,v in sorted(dict_temp.items()):
   #      file.write(str(k)+' '+str(v)+'\n\n')
   #  file.close()

def prehelp2():
    # 声明一个空字典，来保存文本文件数据
    dict_temp = {}

    # 打开文本文件
    file = open('./dev.txt', 'r')

    # 遍历文本文件的每一行，strip可以移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    for line in file.readlines():
        line = line.strip()
        k = line.split(' ')[0]
        v = line.split(' ')[1]
        dict_temp[k] = v

    # 依旧是关闭文件
    file.close()

    #  可以打印出来瞅瞅
    print(dict_temp)
    print(len(dict_temp))
    return  dict_temp;


def prehelp3():
    # 声明一个空字典，来保存文本文件数据
    dict_temp = {}

    # 打开文本文件
    file = open('./dev.txt', 'r')

    # 遍历文本文件的每一行，strip可以移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    for line in file.readlines():
        line = line.strip()
        k = line.split(' ')[0]
        v = line.split(' ')[1]
        dict_temp[k] = v

    # 依旧是关闭文件
    file.close()

    file2 = open('./train.txt', 'r')

    # 遍历文本文件的每一行，strip可以移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    for line in file2.readlines():
        line = line.strip()
        k = line.split(' ')[0]
        v = line.split(' ')[1]
        dict_temp[k] = v

    # 依旧是关闭文件
    file2.close()


    #  可以打印出来瞅瞅
    print(dict_temp)
    print(len(dict_temp))
    return  dict_temp;

def sortedDictValues1(adict):
        items = adict.items()
        items.sort()
        return [value for key, value in items]

def threeToOne():
    dict_temp1 = {}

    # 打开文本文件
    file = open('./eval2_1.txt', 'r')

    # 遍历文本文件的每一行，strip可以移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    for line in file.readlines():
        line = line.strip()
        k = line.split(' ')[0]
        v = line.split(' ')[1]
        dict_temp1[k] = v

    # 依旧是关闭文件
    file.close()

    dict_temp2 = {}

    # 打开文本文件
    file = open('./eval2_2.txt', 'r')

    # 遍历文本文件的每一行，strip可以移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    for line in file.readlines():
        line = line.strip()
        k = line.split(' ')[0]
        v = line.split(' ')[1]
        dict_temp2[k] = v

    # 依旧是关闭文件
    file.close()

    dict_temp3 = {}

    # 打开文本文件
    file = open('./eval2_3.txt', 'r')

    # 遍历文本文件的每一行，strip可以移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    for line in file.readlines():
        line = line.strip()
        k = line.split(' ')[0]
        v = line.split(' ')[1]
        dict_temp3[k] = v

    # 依旧是关闭文件
    file.close()
    dic_temp4 = {}

    for k, v in dict_temp1.items() :
        count=0;
        v2=dict_temp2[k]
        v3=dict_temp3[k]
        if v == "spoof" :
            count=count+1;
        if v2 == "spoof" :
            count=count+1;
        if v3 == "spoof" :
            count=count+1;

        if count >=2 :
            dic_temp4[k]="spoof"
        else:
            dic_temp4[k]="bonafide"


    reshelp(dic_temp4)






if __name__ == "__main__":
    threeToOne()
