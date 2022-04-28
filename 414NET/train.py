import os
import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.nn.functional as F
import torch.utils.data as D
import numpy as np
import jsonx
import argparse
from tqdm import tqdm
from model import DatasetGenerator, SegmentNet

'''
    By Yee Lin solo effort
'''

# Train model process
'''
    @article{qi2016pointnet,
        title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
        author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
        journal={arXiv preprint arXiv:1612.00593},
        year={2016}
    }
'''
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='', help='Saved Model')
parser.add_argument('--dataset', type=str, required=True, help='Dataset Path')
parser.add_argument('--object', type=str, default='Chair', help='Object Type')

args = parser.parse_args()
batch_size = 32
epochs = 100
workers = 4

# Generate the train set and the test set
train_set = DatasetGenerator(path=args.dataset, object_type=args.object)
test_set = DatasetGenerator(path=args.dataset, train_flag=False, object_type=args.object)

# Load data set to PyTorch dataloader
train_dataloader = D.DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=workers,
    shuffle=True    
)

test_dataloader = D.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers
)

print('Arguments:', args)
nums_seg_dict = train_set.num_seg_dict
print('Number of segmentation', nums_seg_dict)
model = SegmentNet(out=nums_seg_dict)

if args.model:
    model.load_state_dict(torch.load(args.model))

# We use Adam algorithm as optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.5)
num_batch = len(train_set)/batch_size
evaluation = []

try:
    os.makedirs('segmentation')
except:
    pass

model.cuda()


'''
    @article{qi2016pointnet,
        title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
        author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
        journal={arXiv preprint arXiv:1612.00593},
        year={2016}
    }
'''
# Train
for epoch in range(epochs):
    scheduler.step()
    for index, data in enumerate(train_dataloader, 0):
        optimizer.zero_grad()
        points, target = data
        points = points.transpose(2, 1)
        points = points.cuda()
        target = target.cuda()
        model = model.train()
        predict, _ = model(points)

        predict = predict.view(-1, nums_seg_dict)
        target = target.view(-1, 1)[:, 0] - 1

        # Caculate the loss function
        loss = F.nll_loss(predict, target)
        loss.backward()
        optimizer.step()
        result = predict.data.max(1)[1]
        correct = result.eq(target.data).cpu().sum()
        print('Train Process: %d | %d: %d' % (epoch, index, num_batch))
        print('Loss %f' % loss.item())
        print('Accuracy %f', correct.item()/float(batch_size * 2500))

        # Test accuracy every 10-batch run
        if index % 10 == 0:
            _, data = next(enumerate(test_dataloader, 0))
            points, target = data
            points = points.transpose(2, 1)
            points = points.cuda()
            target = target.cuda()
            model = model.eval()
            predict, _ = model(points)
            predict = predict.view(-1, nums_seg_dict)
            target = target.view(-1, 1)[:, 0] - 1
            loss = F.nll_loss(predict, target)
            result = predict.data.max(1)[1]
            correct = result.eq(target.data).cpu().sum()
            print('Test Process')
            print('Loss %f' % loss.item())
            print('Accuracy %f', correct.item()/float(batch_size * 2500))

# Save current model
torch.save(model.state_dict(), 'segmentation/segmentation.pth')


'''
    @article{qi2016pointnet,
        title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
        author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
        journal={arXiv preprint arXiv:1612.00593},
        year={2016}
    }
'''
# Evaluation
for index, data in tqdm(enumerate(test_dataloader, 0)):
    points, target = data
    points = points = points.transpose(2, 1)
    points = points.cuda()
    target = target.cuda()
    model = model.eval()
    predict, _ = model(points)
    result = predict.data.max(2)[1]

    predict_array = result.cpu().data.numpy()
    target_array = target.cpu().data.numpy() - 1

    for shape_index in range(target_array.shape[0]):
        parts = range(nums_seg_dict)
        part_evals = []
        for part in parts:
            I = np.sum(np.logical_and(predict_array[shape_index] == part, 
                            target_array[shape_index] == part))

            U = np.sum(np.logical_or(predict_array[shape_index] == part,
                             target_array[shape_index] == part))
            if U == 0:
                part_eval = 1 
            else: part_eval = I/float(U)

            part_evals.append(part_eval)
        
        evaluation.append(np.mean(part_eval))

print("Accuracy=%f" % np.mean(evaluation))

'''
pred_color = []
point_np = []

color = ['red', 'blue', 'yellow', 'green']
content_color_dict = {}
content_dict = {}
counter  = 0
for index in range(len(point_np)):
    if tuple(pred_color[index]) not in content_dict:
        content_dict[tuple(pred_color[index])] = []
        content_dict[tuple(pred_color[index])].append(point_np[index].tolist())
        print(type(point_np[index]))
    else:
        content_dict[tuple(pred_color[index])].append(point_np[index].tolist())
print(content_dict.keys())
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=None, azim=None)

for key in content_dict:
    print(key)
    print(type(content_dict[key]))
    print(content_dict[key])
    x = np.array(content_dict[key])[:, 0]
    y = np.array(content_dict[key])[:, 2]
    z = np.array(content_dict[key])[:, 1]

    ax.scatter(x, y, z, color=color[counter])
    counter += 1

plt.title("simple 3D scatter plot")
'''