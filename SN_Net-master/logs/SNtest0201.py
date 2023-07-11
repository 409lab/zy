# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:12:04 2020

@author: success
"""
import torch
import torch.nn as nn
from PIL import Image
from collections import Counter
import os
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, outp, stride):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]

        layers = [
            USConv2d(
                inp, inp, 3, stride, 1, groups=inp, depthwise=True,
                bias=False),
            USBatchNorm2d(inp),
            nn.ReLU6(inplace=True),

            USConv2d(inp, outp, 1, 1, 0, bias=False),
            USBatchNorm2d(outp),
            nn.ReLU6(inplace=True),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)

class Model(nn.Module):
    def __init__(self, num_classes=74, input_size=224):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        init_channel = 32
        self.block_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        self.features = []

        # head
        assert input_size % 32 == 0
        channels = make_divisible(init_channel)
        self.outp = make_divisible(1024)
        first_stride = 2
        self.features.append(
            nn.Sequential(
                USConv2d(
                    3, channels, 3, first_stride, 1, bias=False,
                    us=[False, True]),
                USBatchNorm2d(channels),
                nn.ReLU6(inplace=True))
        )

        # body
        for c, n, s in self.block_setting:
            outp = make_divisible(c)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        DepthwiseSeparableConv(channels, outp, s))
                else:
                    self.features.append(
                        DepthwiseSeparableConv(channels, outp, 1))
                channels = outp

        # avg_pool_size = input_size// 32
        # self.features.append(nn.AvgPool2d(avg_pool_size))
        self.features.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.classifier = nn.Sequential(
            USLinear(self.outp, num_classes, us=[True, False])
        )
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        #x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
        x = self.features(x)
        last_dim = x.size()[1]
        x = x.view(-1, last_dim)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


#导入模型
model=torch.load('E://ZY//B//slimmable_networks-master//logs//best_model.pt')
device = torch.device('cuda')
model = model.to(device)

#定义标签顺序
name=['东方海笋',
      '中国不等蛤',
      '中国蛤蜊',
      '丽文蛤',
      '习见赤蛙螺', 
      '亚洲棘螺', 
      '光滑河蓝蛤', 
      '凸壳肌蛤', 
      '单齿螺', 
      '厚壳贻贝', 
      '可变荔枝螺', 
      '台湾日月贝', 
      '四角蛤蜊',
      '多角荔枝螺', 
      '大竹蛏', 
      '嫁䗩',
      '密鳞牡蛎', 
      '尖紫蛤', 
      '巴非蛤',
      '弓獭蛤', 
      '微黄镰玉螺',
      '扁玉螺', 
      '斑玉螺', 
      '方斑东风螺',
      '日本镜蛤',
      '朝鲜笋螺',
      '杰氏裁判螺',
      '栉孔扇贝',
      '毛蚶', 
      '江户布目蛤', 
      '沟纹鬘螺',
      '波纹巴非蛤', 
      '泥蚶',
      '浅缝骨螺',
      '海湾扇贝',
      '琴文蛤',
      '疣荔枝螺',
      '白带三角口螺', 
      '皮氏蛾螺',
      '皱纹盘鲍', 
      '短文蛤', 
      '短滨螺', 
      '砂海螂', 
      '等边浅蛤', 
      '粒花冠小月螺', 
      '紫彩血蛤', 
      '紫石房蛤',
      '紫贻贝', 
      '红带织纹螺',
      '纵带滩栖螺',
      '纵肋栉纹螺', 
      '织锦芋螺', 
      '缢蛏',
      '翡翠贻贝', 
      '脉红螺',
      '菲律宾蛤仔',
      '薄片镜蛤', 
      '薪蛤', 
      '虾夷扇贝', 
      '褶纹肋扇贝', 
      '西施舌', 
      '角蝾螺', 
      '角螺', 
      '金刚衲螺', 
      '钝缀锦蛤', 
      '银口凹螺', 
      '锈凹螺', 
      '锥螺',
      '长尾纺锤螺', 
      '长牡蛎', 
      '长竹蛏', 
      '青蛤', 
      '香螺', 
      '魁蚶',
      ]

print(name)
total_num = 0
right_num = 0
total=0
right_totnum=0
error_totnum=0
error_list = []
predicted_list=[]
transform=transforms.Compose([
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                                 std=[0.5,0.5,0.5])
                            ])
#指定文件夹，对其下的文件进行输入然后预测
file_names = []
file_home = os.listdir('D://ZY//DataEnhance0709test')
for i in file_home:
    file_names.append(os.listdir('D://ZY//DataEnhance0709test//'+i))
#print(file_names)
for i in range(len(file_names)):
    #print(len(file_names[i]))#每个种类的图片数量
    total+=len(file_names[i])
    for file_name in file_names[i]:
        #print(file_home[i])#厚壳贻贝
        #print(file_name)#厚壳贻贝 (1).jpg
        total_num=len(file_names[i])
        #fp = open('D://ZY//test set//'+file_home[i]+'//'+file_name,'rb')
        #img = Image.open(fp)
        img=Image.open('D://ZY//DataEnhance0709test//'+file_home[i]+'//'+file_name)
        #print('D://ZY//test0315//'+file_home[i]+'//'+file_name ,  file_home[i])#图片具体路径
        #print('原图：')
        #print(img.size)
        #x.squeeze().size()	# 不加参数，去掉所有为元素个数为1的维度
        #torch.squeeze(x, 0).size()	# 加上参数，去掉第一维的元素，不起作用，因为第一维有2个元素
        #torch.squeeze(x, 1).size()	# 加上参数，去掉第二维的元素，正好为 1，起作用
        im_chang = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.CenterCrop(min(img.size)),
            transforms.Resize((224,224)),
        ])
        im_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.Resize((224,224)),
        ])
        if max(img.size)/min(img.size)>=1.7:
            img=im_chang(img)
        else:
            img=im_aug(img)
            
        img_x=transform(img).unsqueeze(0)
        #print('预处理：')
        #print(img_x.shape)
        img_x = img_x.to(device)
        #img= img.to(device)
        #outputs=model(img)
        outputs = model(img_x)
        print(outputs)
        _, predicted = torch.max(outputs, 1)#每行最大值
        
        a, idx = torch.sort(outputs)
        idx.reshape(1,74)
        print(idx)
        #print(idx[0][52],idx[0][51],idx[0][50])
        print(name[idx[0][73]],name[idx[0][72]],name[idx[0][71]])
        predicted_list.append(name[predicted]) 
        #if (name[predicted]==file_home[i])|(name[idx[0][51]]==file_home[i])|(name[idx[0][50]]==file_home[i]):
        if (name[predicted]==file_home[i]):
            right_num=right_num+1
            right_totnum=right_totnum+1
        else:
            error_list.append(file_name+"==>"+name[predicted])
            #print(error_list)
            error_totnum=error_totnum+1
            print('原图：')
            print(img.size)
            #plt.imshow(img)
            #plt.show()
            print('把'+file_name+'识别成了'+name[predicted])
            
            #if not os.path.exists("E://ZY//分错1020P"):
                #os.mkdir("E://ZY//分错1020P")
            #if not (os.path.exists("E://ZY//分错1020P//%s %s" %(file_home[i],name[predicted]))):
                #os.mkdir("E://ZY//分错1020P//%s %s" %(file_home[i],name[predicted]))
            #img.save("E://ZY//分错1012//%s %s//%i.jpg" %(file_home[i],name[predicted],i))
            #img.save(os.path.join("E://ZY//分错1020P//%s %s" %(file_home[i],name[predicted]), file_name + ".jpg"))
           # if (os.path.exists("E://ZY//分错1020P//%s %s//%i.jpg" %(file_home[i],name[predicted],i))):
                #continue
            
    print(file_home[i]+'的识别正确率为：')
    if total_num !=0:
        print(right_num/total_num)
    else:
        print('100%')
    right_num=0
    Counter(predicted_list)
print('总的准确率：')
print(right_totnum)
print(error_totnum)
print(total)
print(right_totnum/total)