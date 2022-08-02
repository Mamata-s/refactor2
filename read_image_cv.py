import cv2




output_edges = cv2.imread('test_set/preds_edges_4/hr_f4_149_z_78.png')
output_edges = cv2.cvtColor(output_edges, cv2.COLOR_BGR2GRAY)
window_name = 'output_edges'
cv2.imshow(window_name, output_edges)



# input_edges = cv2.imread('test_set/input_edges_4/hr_f4_149_z_212.png')
# input_edges = cv2.cvtColor(input_edges, cv2.COLOR_BGR2GRAY)
# window_name = 'input_edges'
# cv2.imshow(window_name, input_edges)


# label_edges = cv2.imread('test_set/label_edges_2/hr_f4_149_z_20.png')
# label_edges = cv2.cvtColor(label_edges, cv2.COLOR_BGR2GRAY)
# window_name = 'label_edges'
# cv2.imshow(window_name, label_edges)




cv2.waitKey(0) 
cv2.destroyAllWindows() 