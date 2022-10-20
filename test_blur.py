import cv2

# path 
path = r'test_set/hr_f1_160_z_107.png'
  
# Reading an image in default mode 
image = cv2.imread(path) 
  
  
# mean filtering 
ksize = (3, 3)  
mean = cv2.blur(image, ksize, cv2.BORDER_DEFAULT) 
print(mean.shape)

# median filtering
#median blurring
print(type(image))
median = cv2.medianBlur(image,3)
print(median.shape)

cv2.imwrite('mean.png', mean)

