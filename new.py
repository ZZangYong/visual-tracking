import imageio
import os

path = 'C:/record_frames/'
filenames = []
images = []
for frame in os.listdir(path):  # 读取文件夹中的文件
    if os.path.splitext(frame)[1] == '.png':  # 如果后缀名为png
        filenames.append(os.path.join(path, frame))  # 则将其路径加入frame_list中

for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('gif.gif', images, 'GIF', duration=0.01)
