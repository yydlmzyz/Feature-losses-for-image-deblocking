import os
from PIL import Image
import numpy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import mymodel
import myutils


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.input_dir = os.path.join(root_dir,'input')
        self.label_dir = os.path.join(root_dir,'label')
        self.transform = transform

    def __len__(self):
        return os.listdir(self.input_dir).__len__()

    def __getitem__(self, idx):
        input_names = sorted(os.listdir(self.input_dir))
        label_names = sorted(os.listdir(self.label_dir))

        input_name = os.path.join(self.input_dir,input_names[idx])
        input_image =Image.open(input_name)#Image get jpg

        label_name = os.path.join(self.label_dir,label_names[idx])
        label_image =Image.open(label_name)

        sample = {'input_image': input_image, 'label_image': label_image, 'name': input_names[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class mytransform(object):
    def __call__(self, sample):
        input_image, label_image,name= sample['input_image'], sample['label_image'],sample['name']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input_image = numpy.asarray(input_image).transpose(2, 0, 1)/255.0
        label_image = numpy.asarray(label_image).transpose(2, 0, 1)/255.0

        return {'input_image': torch.from_numpy(input_image).float(),
                'label_image': torch.from_numpy(label_image).float(),
                'name':name}
            

def wrap_variable(input_batch, label_batch, use_gpu):
        if use_gpu:
            input_batch, label_batch = (Variable(input_batch.cuda()), Variable(label_batch.cuda()))
        else:
            input_batch, label_batch = (Variable(input_batch),Variable(label_batch))
        return input_batch, label_batch


def checkpoint(name,psnr1,psnr2,ssim1,ssim2):
    print('{},psnr:{:.4f}->{:.4f},ssim:{:.4f}->{:.4f}'.format(name,psnr1,psnr2,ssim1,ssim2))
    #write to text
    output = open(os.path.join(Image_folder,'test_result.txt'),'a+')
    output.write(('{} {:.4f} {:.4f} {:.4f} {:.4f}'.format(name,psnr1,psnr2,ssim1,ssim2))+'\r\n')
    output.close()

def save(output_image,name):
    output_data=output_image.data[0]
    if use_gpu:
        img=255.0*output_data.clone().cpu().numpy()
    else:
        img=255.0*output_data.clone().numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(os.path.join(Image_folder,'output','{}.jpg'.format(name[:-4])))


def test():
    model.eval()
    #input and label
    avg_psnr1 = 0
    avg_ssim1 = 0
    #output and label
    avg_psnr2 = 0
    avg_ssim2 = 0

    for i, sample in enumerate(dataloader):
        input_image,label_image,name=sample['input_image'],sample['label_image'],sample['name'][0]#tuple to str
 
        #Wrap with torch Variable
        input_image,label_image=wrap_variable(input_image,label_image, use_gpu)
        #predict
        output_image = model(input_image)

        #calculate psnr
        psnr1 =myutils.psnr(input_image, label_image)
        psnr2 =myutils.psnr(output_image, label_image)
        # ssim is calculated with the normalize (range [0, 1]) image
        ssim1 = torch.sum((myutils.ssim(input_image, label_image, size_average=False)).data)/1#batch_size
        ssim2 = torch.sum((myutils.ssim(output_image, label_image, size_average=False)).data)/1

        avg_ssim1 += ssim1
        avg_psnr1 += psnr1
        avg_ssim2 += ssim2
        avg_psnr2 += psnr2

        #save output and record
        checkpoint(name,psnr1,psnr2,ssim1,ssim2)
        save(output_image,name)

    #print and save
    avg_psnr1 = avg_psnr1/len(dataloader)
    avg_ssim1 = avg_ssim1/len(dataloader)
    avg_psnr2 = avg_psnr2/len(dataloader)
    avg_ssim2 = avg_ssim2/len(dataloader)

    print('Avg. PSNR: {:.4f}->{:.4f} Avg. SSIM: {:.4f}->{:.4f}'.format(avg_psnr1,avg_psnr2,avg_ssim1,avg_ssim2))
    output = open(os.path.join(Image_folder,'test_result.txt'),'a+')
    output.write('Avg. PSNR: {:.4f}->{:.4f} Avg. SSIM: {:.4f}->{:.4f}'.format(avg_psnr1,avg_psnr2,avg_ssim1,avg_ssim2)+'\r\n')
    output.close()


#------------------------------------------------------------------
#cuda
use_gpu=torch.cuda.is_available()

#set path
root_dir=os.getcwd()
Image_folder=os.path.join(root_dir,'test')
model_weights_file=os.path.join(root_dir,'12-0.0031-0.0030-25.7129-25.3611param.pth')

#set model  
model=mymodel.L8()
if use_gpu:
    model = model.cuda()
model.load_state_dict(torch.load(model_weights_file))

#set dataset and dataloader
mydataset = ImageDataset(root_dir=Image_folder, transform=mytransform())
dataloader = DataLoader(mydataset, batch_size=1,shuffle=False, num_workers=0)



def main():
    test()

if __name__=='__main__':
    main()
