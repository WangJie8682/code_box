import numpy as np
def im2col(image,k_size,stride):
    #image shape [B,W,H,C]
    imgcol = []#[B,(H-k+1)*(W-k+1),k*K*C_input]
    for b in range(image.shape[0]):
        for i in range(0,image.shape[1]-ksize+1,stride):
            for j in range(0,image.shape[2]-ksize+1,stride):
                col = image[b,i:i+ksize,j:j+ksize,:].reshape([-1])
                imgcol.append(col)
    imgcol = np.array(imgcol)
    return imgcol


class Conv2D(object):
    def __init__(self,output_channels,stride = 1,ksize = 3,method = 'VALID'):
        self.output_channels = output_channels
        self.ksize = ksize
        self.stride = stride
        self.method = method

    def Outshape(self,shape):
        self.input_shape = shape
        self.input_channels = shape[-1]
        weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / self.output_channels)
        self.weights = np.random.standard_normal((self.ksize*self.ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        if (shape[1] - self.ksize) % self.stride != 0:
            print('input tensor width can\'t fit stride')
        if (shape[2] - self.ksize) % self.stride != 0:
            print('input tensor height can\'t fit stride')
        if self.method == 'VALID':
            return [shape[0],
                    (shape[1] - self.ksize + 1) // self.stride,
                    (shape[1] - self.ksize + 1) // self.stride,
                    self.output_channels]
        return [shape[0],
                shape[1]// self.stride,
                shape[2]// self.stride,
                self.output_channels]


    def forward(self,x):
        shape = x.shape
        if self.method == "VALID":
            self.out = np.zeros((shape[0],int((shape[1]-self.ksize+1)/self.stride),int((shape[2]-self.ksize+1)/self.stride),self.output_channels))
        if self.method == "same":
            self.out = np.zeros((shape[0],shape[1]//self.stride,shape[2]//self.stride,self.output_channels))
        if self.method == "same":
            x = np.pad(x,((0,0),(self.ksize//2,self.ksize//2),(self.ksize//2,self.ksize//2),(0,0)),'constant',constant_values= 0)
        col_weights = self.weights.reshape([-1,self.output_channels])#[k*k*Cinput,Coutput]
        self.col_image = []
        self.batchsize = shape[0]
        conv_out = np.zeros(self.out.shape)
        self.col_image = im2col(x,ksize,stride)
        conv_out = np.dot(self.col_image,col_weights)
        conv_out= np.reshape(conv_out,self.out.shape)
        return conv_out

    def backward(self,out):
        self.out = out
        col_out = np.reshape(out,[self.batchsize,-1,self.output_channels])
        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T,col_out[i]).reshape(self.weights.shape)#KC = XC.T * YC
        self.b_gradiend += np.sum(col_out,axis = (0,1))

        #deconv
        if self.method == "VALID":
            pad_out = np.pad(self.out,((0,0),(self.ksize-1,self.ksize-1),(self.ksize-1,self.ksize-1),(0,0),'constant',constant_values = 0))
        if self.method == "same":
            pad_out = np.pad(self.out,((0,0),(self.ksize//2,self.ksize//2),(self.ksize//2,self.ksize//2),(0,0),'constant',               constant_values = 0))
        flip_weights = self.weights[::-1,...]#flipud（m）等价于m[::-1,...]，等价于flip(m,0)，而fliplr（m）等价于m[:,::-1,...],等价于flip(m,1)。
        flip_weights = flip_weights.swapaxes(1,2)
        col_flip_weights = flip_weights.reshape([-1,self.input_channels])
        col_pad_out = im2col(pad_out,self.ksize,self.stride)
        next_out = np.dot(col_pad_out,col_flip_weights)
        next_out = np.reshape(next_out,self.input_shape)
        return next_out

    def gradient(self,alpha = 0.0001,weight_decay = 0.0004):
        self.weights -= alpha*self.w_gradient/self.batchsize
        self.bias -= alpha*self.b_gradiend/self.batchsize

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradiend = np.zeros(self.bias.shape)

