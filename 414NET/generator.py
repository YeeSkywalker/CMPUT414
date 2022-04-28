import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data as D
from torch.autograd import Variable
from train import DatasetGenerator, SegmentNet
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', help='model')
parser.add_argument('--content', type=int, default=0, help='content object')
parser.add_argument('--style', type=int, default=0, help='style object')
parser.add_argument('--dataset', type=str, default='', help='dataset')
parser.add_argument('--object', type=str, default='', help='object type')

args = parser.parse_args()
print(args)

data_set = DatasetGenerator(path=args.dataset, object_type= args.object, train_flag=False)

content_idx = args.content
style_idx = args.style

content_point, content_seg = data_set[content_idx]
print(content_point.size(), content_seg.size())
point_np_content = content_point.numpy()

style_point, style_seg = data_set[style_idx]
print(style_point.size(), style_seg.size())
point_np_style = style_point.numpy()

state_dict = torch.load(args.model)
classifier = SegmentNet(out=state_dict['conv4.weight'].size()[0])
classifier.load_state_dict(state_dict)
classifier.eval()

content_point = content_point.transpose(1, 0).contiguous()
style_point = style_point.transpose(1, 0).contiguous()

content_point = Variable(content_point.view(1, content_point.size()[0], point1.size()[1]))
pred_content, _, _ = classifier(content_point)
pred_choice_content = pred_content.data.max(2)[1]

style_point = Variable(style_point.view(1, style_point.size()[0], style_point.size()[1]))
pred_style, _, _ = classifier(style_point)
pred_choice_style = pred_style.data.max(2)[1]

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]

pred_color_content = cmap[pred_choice_content.numpy()[0], :]
pred_color_style = cmap[pred_choice_style.numpy()[0], :]

content_dict = {}
counter_content  = 0
for index in range(len(point_np_content)):
    if tuple(pred_color_content[index]) not in content_dict:
        content_dict[tuple(pred_color_content[index])] = []
        content_dict[tuple(pred_color_content[index])].append(point_np_content[index].tolist())
    else:
        content_dict[tuple(pred_color_content[index])].append(point_np_content[index].tolist())

style_dict = {}
counter_style  = 0
for index in range(len(point_np_style)):
    if tuple(pred_color_style[index]) not in style_dict:
        style_dict[tuple(pred_color_style[index])] = []
        style_dict[tuple(pred_color_style[index])].append(point_np_style[index].tolist())
    else:
        style_dict[tuple(pred_color_style[index])].append(point_np_style[index].tolist())

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=None, azim=None)

for key in content_dict:
    print(key)
    print(type(content_dict[key]))
    if counter_content != 1 and counter_content != 3:
        x = np.array(content_dict[key])[:, 0]
        y = np.array(content_dict[key])[:, 2]
        z = np.array(content_dict[key])[:, 1]

        ax.scatter(x, y, z, color='blue')
    counter_content += 1

for key in style_dict:
    print(key)
    print(type(style_dict[key]))

    if counter_style == 1 or counter_style == 3:
        x = np.array(style_dict[key])[:, 0]
        y = np.array(style_dict[key])[:, 2]
        z = np.array(style_dict[key])[:, 1]

        ax.scatter(x, y, z, color='blue')
    counter_style += 1

plt.title("Style Transfer Result")
 
# show plot
plt.savefig('output.pdf') 
print('Output')
plt.close()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=None, azim=None)
counter_content = 0
for key in content_dict:
    print(key)
    print(type(content_dict[key]))
    x = np.array(content_dict[key])[:, 0]
    y = np.array(content_dict[key])[:, 2]
    z = np.array(content_dict[key])[:, 1]

    ax.scatter(x, y, z, color='blue')
    counter_content += 1

plt.title("Content Object")
 
# show plot
plt.savefig('content.pdf') 
plt.close()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=None, azim=None)
counter_style = 0
for key in style_dict:
    print(key)
    print(type(style_dict[key]))
    x = np.array(style_dict[key])[:, 0]
    y = np.array(style_dict[key])[:, 2]
    z = np.array(style_dict[key])[:, 1]

    ax.scatter(x, y, z, color='blue')
    counter_style += 1

plt.title("Style Object")
 
# show plot
plt.savefig('content.pdf') 
plt.close()

