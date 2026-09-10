# Will store total number of voids
# Will store dictionary of void index and its volume
# Will store dictionary of void index, slice index, center coordinate, surface area coverd in that slice
# Will annotate bounding box of void with its surface area if (annotate is true)

# Flow:
# call constructor, load images, run, if true at the end will save image (can utilise from manager)
# annotation and text (font size 12?) #1ED760 

# factory (load image, run, annotate, save image), will implement defects later

# can utiise worker threads if find suitable