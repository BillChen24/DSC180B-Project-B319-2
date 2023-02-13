import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def plot_tensor(tensor):
    plt.imshow(tensor.permute(1, 2, 0))
    
def plot_background(bg):
    if bg['day'] == None:
        print("No day images")
    else:
        plt.subplot(1, 2, 1)
        plot_tensor(bg['day'])
        plt.title('daytime background')
        
    if bg['night'] == None:
        print("No night images")
    else:
        plt.subplot(1, 2, 2)
        plot_tensor(bg['night'])
        plt.title('nighttime background')

def with_vs_no_background(tensor, bg):
    plt.figure()
    plt.subplot(1, 2, 1)
    plot_tensor(tensor)
    plt.title('original image')
    plt.subplot(1, 2, 2)
    plot_tensor(tensor - bg)
    plt.title('No background')
        
def plot_scatter(img_path, img_name = "Image", x_axis = "Red", y_axis = "Blue"):
    img = Image.open(img_path)
    arr = np.array(img).astype(np.float64)/256
    arr_shape = arr.shape
    arr_color = arr.reshape(arr_shape[0]*arr_shape[1], arr_shape[2])
    color_dict = {"Red": 0, "Green": 1, "Blue": 2}
    x = arr[:,:,color_dict[x_axis]]
    y = arr[:,:,color_dict[y_axis]]
    plt.scatter(x, y, c = arr_color)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.xticks([])
    plt.yticks([])
    plt.title(img_name)

def histOfAnimal(picList, label):
    mapped = list(zip(picList, label))
    listA = [i for i in mapped if i[1] != 'Unknown']
    #listN = [i for i in mapped if i[1] == 'no']
    #list_im = ["images/1a.jpg", "images/3a.jpg"]
    imgs = [ Image.open(i) for i in listA ]
    #imgsN = [ Image.open(i) for i in listN ]
# pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack([i.resize(min_shape) for i in imgs])
    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save('comba.jpg')
    imageObj = cv2.imread('comba.jpg')

# Get RGB data from image
    blue_color = cv2.calcHist([imageObj], [0], None, [256], [0, 256])
    red_color = cv2.calcHist([imageObj], [1], None, [256], [0, 256])
    green_color = cv2.calcHist([imageObj], [2], None, [256], [0, 256])

