import cv2
import numpy as np
img_path = "/Users/wangjie/Downloads/a.png"
img = cv2.imread(img_path)


def midfilter(image):
    h,w,c = image.shape[0], image.shape[1], image.shape[2]
    image = np.pad(image, mode = 'constant', constant_values = (0), pad_width = ((1,1),(1,1),(0,0)))
    out = np.zeros(shape = image.shape,dtype = np.float32)
    import pdb;pdb.set_trace()
    for i in range(1,h):
        for j in range(1,w):
            for  k in range(c):
                out[i][j][k] = np.mean(image[i-1:i+2,j-1:j+2,k])
    import pdb;pdb.set_trace()
    return out[1:h+1,1:w+1,:]

cv2.imwrite('img.jpg',midfilter(img))
