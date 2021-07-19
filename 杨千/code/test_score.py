tp = 0; tn = 0; fp=0; fn=0;
pre = open('my_best_models/ocsoftmax/dev-result.txt', 'r')
real = open('../zju_deepfake/dev.txt','r')
pre_lines = pre.readlines()
real_lines = real.readlines()
for pre_line, real_line in zip(pre_lines, real_lines):
    pre_line = pre_line.split(' ')
    pre_tag = pre_line[1]
    pre_tag = pre_tag.replace('\n', '')
    real_line = real_line.split(' ')
    real_tag = real_line[1]
    real_tag = real_tag.replace('\n', '')
    if(pre_tag == 'bonafide'):
        if(real_tag == 'bonafide'):
            tp += 1
        else:
            fn += 1
    elif(pre_tag == 'spoof'):
        if(real_tag == 'bonafide'):
            fn += 1
        else:
            tn += 1
    else:
        print("error")
        print(pre_tag)
        exit()
acc = (tp + tn)/(tp + tn + fp + fn)
recall = tp / (tp + fn)
pre = tp / (tp + fp)
print("accuarcy:%lf" % acc)
print("recall:%lf" %recall)
print("precision:%lf" %pre)
