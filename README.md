## Feature losses for image deblocking
### 1.Introduction  
&emsp;&emsp;以前的图像去块方法是在有监督模式下训练一个前馈卷积神经网络，用逐像素差距做损失函数（MSE）来衡量输出图像与目标图像的差距。然后训练网络对图像去块。  
&emsp;&emsp;为进一步提高去块效果，参考：[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)的方法，训练一个CNN去块网络，但不仅以输出图像和目标图像的MSE构造损失函数，而通过一个预训练好的网络vgg16中提取输出图像和目标图像的高级特征，构造新的损失函数来训练网络。  
  
### 2.Model
&emsp;&emsp;图像去块网络暂用[L8](https://arxiv.org/pdf/1605.00366.pdf)。特征提取网络用[vgg16](https://arxiv.org/abs/1409.1556)。参考论文中用来构造损失函数的特征层是其13层卷积层中的第2、4、7、10层卷积的激活输出。实验中暂只用了第4层卷积层的输出。  
&emsp;&emsp;L8和vgg都经过预训练。
### 3.Train
&emsp;&emsp;1.数据集：train[100800,3,42,42],validation约[15000,3,42,42]，test[200,3,481,321]。RGB格式，压缩率10  
&emsp;&emsp;2.训练：batchsize:64 epoch:50 optimizer:Adam(lr=0.001)

### 4.Experiment Result
训练模型 | PSNR|SSIM
---|---|---|
input(Q10) | 25.6128|0.7395|
pretrained model | 26.8396(+1.23)|0.7792(+0.040)
only feature loss*5 |25.7601(+0.15)|0.7304(-0.001)
only pixel loss*1|26.8711(+1.26)|0.7815(+0.042)
both pixel loss*1+ feature loss*5|26.5565(+0.9437)|0.7726(+0.033)


备注：  
 pixel loss是输出图像和目标图像的MSE，约为0.0025-0.0023之间，权重为1  
feature loss是输出图像和目标图像经过vgg第4层卷积输出的特征层的MSE，在0.0008-0.0006之间，权重为5



### 5.Analysis&Question
#### 1.metrics:  
&emsp;&emsp;换成feature loss后，PSNR、SSIM比原来低，这应该是正常的，参考论文中强调了这一点，因为SSIM和PSNR都是基于像素的衡量方式，并不能很好的衡量人类的视觉质量；从实验中也可以看出，虽然使用feature  loss后，PSNR、SSIM降低了，但是人眼却感觉到质量比原来好，主要是感觉到图像更清晰，比如下图：![image](https://github.com/yydlmzyz/Feature-losses-for-image-deblocking/blob/master/test/label/2018.jpg)但这是一种定性的判断，缺乏衡量手段，有待改进

#### 2.model&net:  
&emsp;&emsp;所用的特征提取模型vgg作为一个图像识别模型，它本身也和人眼的视觉功能更相关。即它所提取到的特征是高级特征表示。所以用vgg提取特征损失是合适的。  
&emsp;&emsp;从vgg中选择第4层卷积输出作为特征是直接参考的原论文。因为低层感知结构、纹理等低级特征，高层感知语义、内容等高级特征，图像去块与低级特征更有关系，所以用低层输出构建损失函数是合理的。

#### 3.loss weights:  
&emsp;&emsp;在单独使用feature loss时，pixel loss并没有减小；在单独使用pixel loss式，feature loss也没有明显减小；共同使用时，两者都在减小。两个loss有关系但也有区别。  
&emsp;&emsp;实验中混合两个loss时，权重为1：5，目的是使两种loss的值尽量相同，没有仔细分析其影响，有待改进。

#### 4.new problem:  
&emsp;&emsp;虽然输出图像的块效应有改善，但是却产生了噪声，尤其是在白色区域，更多更明显：

![image](http://note.youdao.com/favicon.ico)
以前并没有出现这种问题，但这次很严重，还没有解决，猜测其原因可能是由于训练不足、也有可能是与使用RGB格式有关，以前用YCbCr格式的时候没出现这种问题、也有可能是数据处理上有问题。有待解决。

#### 5.speed
&emsp;&emsp;参考论文中提到用feature loss训练有加速收敛的作用，由于用了预训练好的模型，并没有观察到，有待测量
  
### References  
1.[pythoch examples fast neural style](https://github.com/pytorch/examples/tree/master/fast_neural_style)  
2.[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)  
3.[Compression Artifacts Removal Using
Convolutional Neural Networks](https://arxiv.org/pdf/1605.00366.pdf)

