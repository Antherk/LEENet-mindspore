import glob
import mindspore as ms
import re
from PIL import Image
from mindspore import ops
from dataload import find_image_file


class leenet_loader:
    def __init__(self, images_path):
        self.train_list = find_image_file(images_path) 
        self.size = 512
        self.data_list = self.train_list

    def __getitem__(self, index):
        zero_image_path = self.data_list[index]
        all_image_path = glob.glob(zero_image_path.replace('_000.','_*.'))
        all_image_path.sort(key=function)
        zeros = ops.Zeros()
        output = zeros((len(all_image_path),3,self.size,self.size), ms.float32) #创建一个用于存放图片数据的全零tensor
        label = zeros((len(all_image_path)))
        image_num = 0
        pattern = re.compile(r'\d+')
        for each_image_path in all_image_path:
            label_each = int(pattern.findall(each_image_path)[-1])
            label[image_num] = 1+label_each/10.0
            each_image = Image.open(each_image_path)
            #each_image = numpy_to_tensor(each_image,self.size)
            each_image = ms.Tensor(each_image)  #numpy to mindspore_Tensor
            output[image_num,:,:,:] = each_image
            image_num += 1
        return output,label
    
    def __len__(self):
        return len(self.data_list)