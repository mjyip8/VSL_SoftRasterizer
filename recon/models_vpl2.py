import torch
import torch.nn as nn
import torch.nn.functional as F

import soft_renderer as sr
import soft_renderer.functional as srf
import math

class VPLEncoder(nn.Module):
    def __init__(self, dim_in=4, dim_out=512):
        super(VPLEncoder, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        dim_h = 64
        self.conv_in = nn.Conv2d(dim_in, dim_h * 1, 7, stride=2, padding=3, bias=False)
        self.conv1_1_1 = nn.Conv2d(dim_h * 1, dim_h * 1, 3, stride=1, padding=1, bias=False)
        self.conv1_1_2 = nn.Conv2d(dim_h * 1, dim_h * 1, 3, stride=1, padding=1, bias=False)
        self.conv1_2_1 = nn.Conv2d(dim_h * 1, dim_h * 1, 3, stride=1, padding=1, bias=False)
        self.conv1_2_2 = nn.Conv2d(dim_h * 1, dim_h * 1, 3, stride=1, padding=1, bias=False)

        self.conv2_1_1 = nn.Conv2d(dim_h * 1, dim_h * 2, 3, stride=2, padding=1, bias=False)
        self.conv2_1_2 = nn.Conv2d(dim_h * 2, dim_h * 2, 3, stride=1, padding=1, bias=False)
        self.conv2_1_3 = nn.Conv2d(dim_h * 1, dim_h * 2, 1, stride=2, padding=0, bias=False)
        self.conv2_2_1 = nn.Conv2d(dim_h * 2, dim_h * 2, 3, stride=1, padding=1, bias=False)
        self.conv2_2_2 = nn.Conv2d(dim_h * 2, dim_h * 2, 3, stride=1, padding=1, bias=False)
        
        self.conv3_1_1 = nn.Conv2d(dim_h * 2, dim_h * 4, 3, stride=2, padding=1, bias=False)
        self.conv3_1_2 = nn.Conv2d(dim_h * 4, dim_h * 4, 3, stride=1, padding=1, bias=False)
        self.conv3_1_3 = nn.Conv2d(dim_h * 2, dim_h * 4, 1, stride=2, padding=0, bias=False)
        self.conv3_2_1 = nn.Conv2d(dim_h * 4, dim_h * 4, 3, stride=1, padding=1, bias=False)
        self.conv3_2_2 = nn.Conv2d(dim_h * 4, dim_h * 4, 3, stride=1, padding=1, bias=False)

        self.conv4_1_1 = nn.Conv2d(dim_h * 4, dim_h * 8, 3, stride=2, padding=1, bias=False)
        self.conv4_1_2 = nn.Conv2d(dim_h * 8, dim_h * 8, 3, stride=1, padding=1, bias=False)
        self.conv4_1_3 = nn.Conv2d(dim_h * 4, dim_h * 8, 1, stride=2, padding=0, bias=False)
        self.conv4_2_1 = nn.Conv2d(dim_h * 8, dim_h * 8, 3, stride=1, padding=1, bias=False)
        self.conv4_2_2 = nn.Conv2d(dim_h * 8, dim_h * 8, 3, stride=1, padding=1, bias=False)

#         self.linear_out = nn.Linear(dim_h * 8 * 7 * 7, dim_out)
        self.linear_out = nn.Linear(dim_h * 8 * 2 * 2, dim_out) # 2048 for 64x64 pixel images
#         self.linear_out = nn.Linear(2, dim_out)

        self.linear_in_bn = nn.BatchNorm2d(dim_h * 1)
#         self.linear_in_bn = nn.BatchNorm1d(dim_h * 1)
    
        self.conv_1_1_2_bn = nn.BatchNorm2d(dim_h * 1)
        self.conv_1_2_1_bn = nn.BatchNorm2d(dim_h * 1)
        self.conv_1_2_2_bn = nn.BatchNorm2d(dim_h * 1)
        self.conv_2_1_1_bn = nn.BatchNorm2d(dim_h * 1)
        self.conv_2_1_2_bn = nn.BatchNorm2d(dim_h * 2)
        self.conv_2_2_1_bn = nn.BatchNorm2d(dim_h * 2)
        self.conv_2_2_2_bn = nn.BatchNorm2d(dim_h * 2)
        self.conv_3_1_1_bn = nn.BatchNorm2d(dim_h * 2)
        self.conv_3_1_2_bn = nn.BatchNorm2d(dim_h * 4)
        self.conv_3_2_1_bn = nn.BatchNorm2d(dim_h * 4)
        self.conv_3_2_2_bn = nn.BatchNorm2d(dim_h * 4)
        self.conv_4_1_1_bn = nn.BatchNorm2d(dim_h * 4)
        self.conv_4_1_2_bn = nn.BatchNorm2d(dim_h * 8)
        self.conv_4_2_1_bn = nn.BatchNorm2d(dim_h * 8)
        self.conv_4_2_2_bn = nn.BatchNorm2d(dim_h * 8)
        self.linear_out_bn = nn.BatchNorm2d(dim_h * 8)
#         self.linear_out_bn = nn.BatchNorm1d(dim_h * 8)
#         self.linear_out_bn = nn.BatchNorm1d(2048)

    def forward(self, x):
        # [224 -> 112]
        h = F.relu(self.linear_in_bn(self.conv_in(x)))
        h = F.max_pool2d(h, kernel_size=3, stride=2, padding=0)

        # [112 -> 56]
        h1 = self.conv1_1_1(h)
        h1 = self.conv1_1_2(F.relu(self.conv_1_1_2_bn(h1)))
        h = h + h1
        h1 = self.conv1_2_1(F.relu(self.conv_1_2_1_bn(h)))
        h1 = self.conv1_2_2(F.relu(self.conv_1_2_2_bn(h1)))
        h = h + h1

        # [56 -> 28]
        h1 = self.conv2_1_1(F.relu(self.conv_2_1_1_bn(h)))
        h1 = self.conv2_1_2(F.relu(self.conv_2_1_2_bn(h1)))
        h2 = self.conv2_1_3(h)
        h = h1 + h2
        h1 = self.conv2_2_1(F.relu(self.conv_2_2_1_bn(h)))
        h1 = self.conv2_2_2(F.relu(self.conv_2_2_2_bn(h1)))
        h = h + h1

        # [28 -> 14]
        h1 = self.conv3_1_1(F.relu(self.conv_3_1_1_bn(h)))
        h1 = self.conv3_1_2(F.relu(self.conv_3_1_2_bn(h1)))
        h2 = self.conv3_1_3(h)
        h = h1 + h2
        h1 = self.conv3_2_1(F.relu(self.conv_3_2_1_bn(h)))
        h1 = self.conv3_2_2(F.relu(self.conv_3_2_2_bn(h1)))
        h = h + h1

        # [14 -> 7]
        h1 = self.conv4_1_1(F.relu(self.conv_4_1_1_bn(h)))
        h1 = self.conv4_1_2(F.relu(self.conv_4_1_2_bn(h1)))
        h2 = self.conv4_1_3(h)
        h = h1 + h2
        h1 = self.conv4_2_1(F.relu(self.conv_4_2_1_bn(h)))
        h1 = self.conv4_2_2(F.relu(self.conv_4_2_2_bn(h1)))
        h = h + h1
#         print(h.size)

        # [7 -> 1]
        h = h.view(x.size(0), -1)
#         h = F.relu(self.linear_out_bn(h))
        h = F.relu(h)
        h = self.linear_out(h)

        return h

# https://github.com/hiroharu-kato/view_prior_learning/blob/master/mesh_reconstruction/decoders.py
class VPLDecoder(nn.Module):
    def __init__(self, filename_obj, dim_in=512, centroid_scale=0.1, bias_scale=1.0):
        super(VPLDecoder, self).__init__()

        # self.vertices_base, self.faces = neural_renderer.load_obj(filename_obj)
        # self.num_vertices = self.vertices_base.shape[0]
        # self.num_faces = self.faces.shape[0]
        # self.obj_scale = 0.5
        # self.object_size = 1.0
        # self.scaling = scaling

        # dim_hidden = [4096, 4096]
        # init = chainer.initializers.HeNormal()
        # self.linear1 = cl.Linear(dim_in, dim_hidden[0], initialW=init)
        # self.linear2 = cl.Linear(dim_hidden[0], dim_hidden[1], initialW=init)
        # self.linear_bias = cl.Linear(dim_hidden[1], self.num_vertices * 3, initialW=init)

        # self.laplacian = get_graph_laplacian(self.faces, self.num_vertices)

        self.template_mesh = sr.Mesh.from_obj(filename_obj)
        self.register_buffer('vertices_base', self.template_mesh.vertices.cpu()[0]) # vertices_base
        self.register_buffer('faces', self.template_mesh.faces.cpu()[0]) # faces

        self.num_vertices = self.vertices_base.size(0)
        self.num_faces = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5

        dim_hidden = [4096, 4096]
        self.linear1 = nn.Linear(dim_in, dim_hidden[0])
        self.linear2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.linear_bias = nn.Linear(dim_hidden[1], self.num_vertices * 3)
#         self.fc_centroid = nn.Linear(dim_hidden[1], 3)
        self.fc_centroid = nn.Linear(512, 3)

    def forward(self,x):
        batch_size = x.shape[0]

        h = F.relu(self.linear1(x))
        h = F.relu(self.linear2(h))
        bias = self.linear_bias(h) * self.bias_scale
        bias = bias.view(-1, self.num_vertices, 3)

        base = self.vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

        centroid = self.fc_centroid(x) * self.centroid_scale
        
        centroid = torch.tanh(centroid[:, None, :])
        scale_pos = 1 - centroid
        scale_neg = centroid + 1

        vertices = torch.sigmoid(base + bias) * sign
        vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        vertices = vertices + centroid
        vertices = vertices * 0.5
        faces = self.faces[None, :, :].repeat(batch_size, 1, 1)
        
        return vertices, faces

class Model(nn.Module):
    def __init__(self, filename_obj, args):
        super(Model, self).__init__()

        self.encoder = VPLEncoder()
        self.decoder = VPLDecoder(filename_obj)
        self.renderer = sr.SoftRenderer(image_size=args.image_size, sigma_val=args.sigma_val, 
                                        aggr_func_rgb='hard', camera_mode='look_at', viewing_angle=15,
                                        dist_eps=1e-10)
        self.laplacian_loss = sr.LaplacianLoss(self.decoder.vertices_base, self.decoder.faces)
        self.flatten_loss = sr.FlattenLoss(self.decoder.faces)

    def model_param(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def set_sigma(self, sigma):
        self.renderer.set_sigma(sigma)

    def reconstruct(self, images):
        vertices, faces = self.decoder(self.encoder(images))
        return vertices, faces

    def predict_multiview(self, image_a, image_b, viewpoint_a, viewpoint_b):
        batch_size = image_a.size(0)
        # [Ia, Ib]
        images = torch.cat((image_a, image_b), dim=0)
        # [Va, Va, Vb, Vb], set viewpoints
        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)
        self.renderer.transform.set_eyes(viewpoints)

        vertices, faces = self.reconstruct(images)
        laplacian_loss = self.laplacian_loss(vertices)
        flatten_loss = self.flatten_loss(vertices)

        # [Ma, Mb, Ma, Mb]
        vertices = torch.cat((vertices, vertices), dim=0)
        faces = torch.cat((faces, faces), dim=0)

        # [Raa, Rba, Rab, Rbb], cross render multiview images
        print(vertices.shape)
        print(faces.shape)
        silhouettes = self.renderer(vertices, faces)
        return silhouettes.chunk(4, dim=0), laplacian_loss, flatten_loss
        # TODO: Add normal_loss, edge_length_regularization losses
        # For these, we can feed the images view that is used as in train.py

    def evaluate_iou(self, images, voxels):
        vertices, faces = self.reconstruct(images)

        faces_ = srf.face_vertices(vertices, faces).data
        faces_norm = faces_ * 1. * (32. - 1) / 32. + 0.5
        voxels_predict = srf.voxelization(faces_norm, 32, False).cpu().numpy()
        voxels_predict = voxels_predict.transpose(0, 2, 1, 3)[:, :, :, ::-1]
        iou = (voxels * voxels_predict).sum((1, 2, 3)) / (0 < (voxels + voxels_predict)).sum((1, 2, 3))
        return iou, vertices, faces

    def forward(self, images=None, viewpoints=None, voxels=None, task='train'):
        if task == 'train':
            return self.predict_multiview(images[0], images[1], viewpoints[0], viewpoints[1])
        elif task == 'test':
            return self.evaluate_iou(images, voxels)
