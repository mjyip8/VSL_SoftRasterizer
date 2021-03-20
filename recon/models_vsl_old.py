import torch
import torch.nn as nn
import torch.nn.functional as F

import soft_renderer as sr
import soft_renderer.functional as srf
import math

global_latent_dim  = 5
local_latent_dim   = 2
local_latent_num   = 3

def _sampling(self, z_mean, z_logstd, latent_dim, batch_size):
    epsilon = torch.randn((self.batch_size, latent_dim))
    return z_mean + torch.exp(z_logstd) * epsilon

class InfModel(nn.Module):
    def __init__(self, dim_in=4, dim_out=512, dim1=32, dim2=256, im_size=64):
        super(VSL_InfModel, self).__init__()
        self.im_size = im_size
        dim_hidden = [dim1, dim1*2, dim1*4, dim2, 100]
        dim_latent = 100

        self.conv1 = nn.Conv3d(dim_in, dim_hidden[0], kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv3d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.Conv3d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=2, padding=0)

        self.bn1 = nn.BatchNorm3d(dim_hidden[0])
        self.bn2 = nn.BatchNorm3d(dim_hidden[1])
        self.bn3 = nn.BatchNorm3d(dim_hidden[2])
        
        self.z_mean_fc = nn.Linear(dim_out, im_size)
        self.z_logstd_fc = nn.Linear(dim_out, im_size)
        self.zzi_fc1 = nn.Linear(im_size, im_size)
        self.zzi_fc1 = nn.Linear(im_size, im_size)
        self.allzi_fc1 = nn.ModuleList([nn.Linear(500, im_size) for _ in range(self.local_latent_num-1)])
        self.allzi_fc2 = nn.ModuleList([nn.Linear(500, im_size) for _ in range(self.local_latent_num)])
        self.zi_mean = nn.ModuleList([nn.Linear(im_size, self.local_latent_dim) for _ in range(self.local_latent_num)])
        self.zi_logstd = nn.ModuleList([nn.Linear(im_size, self.local_latent_dim) for _ in range(self.local_latent_num)])

    def forward(self, x):
        enc_fc2, z_mean, z_logstd, z_all, k1_loss = (torch.zeros((100, self.local_latent_num + 1)) for _ in range(5))
        # input_shape x -> local_lat z_i
        for i in range(self.local_latent_num + 1):
            tmp = F.relu(self.bn1(self.conv1[i](x)), inplace=True)
            tmp = F.relu(self.bn2(self.conv2[i](tmp)), inplace=True)
            tmp = F.relu(self.bn3(self.conv3[i](tmp)), inplace=True)
            tmp = tmp.view(tmp.size(0), -1)
            tmp = F.relu(self.fc1[i](tmp), inplace=True)
            enc_fc2[i] = F.relu(self.fc2[i](tmp), inplace=True)
        
        # sample global latent variable
        z_mean[0] = self.z_mean_fc(enc_fc2)
        z_logstd[0] = self.z_logstd_fc(enc_f2)
        z_all[0] = self._sampling(self.z_mean[0], self.z_logstd[0], self.global_latent_dim)
        
        enc_zzi_fclayer2, enc_allzi_fclayer2 = (torch.zeros((100, self.local_latent_num) for _ in range(2))
        enc_zizi_fclayer2 = torch.zeros((self.im_size, self.local_latent_num - 1) 

        for i in range(self.local_latent_num):
            # z -> z_i
            tmp = F.relu(self.zzi_fc1(z_all[0]))
            enc_zzi_fclayer2[i] = F.relu(self.zzi_fc2(tmp))
            
            if i == 0:  # sampling z_1
                tmp = torch.cat([enc_zzi_fclayer2[i], enc_fclayer2[i+1]], axis=1)
                tmp = self.allzi_fclayer1[i](tmp) 
                enc_allzi_fclayer2[i] = self.allzi_fclayer2[i](tmp) 

                z_mean[1] = self.zi_mean[i](enc_allzi_fclayer2[i]) 
                z_logstd[1] = self.zi_logstd[i](enc_allzi_fclayer2[i]) 
                z_all[1] = self.sampling(z_mean[1], z_logstd[1], self.local_latent_dim) 
            else:   # sampling z_i (i >= 1)   
                tmp = F.relu(self.zizi_fc1[i-1](z_all[i]))
                enc_zizi_fclayer2[i - 1] = F.relu(tmp)
             
                tmp = torch.cat([enc_zzi_fclayer2[i], enc_fclayer2[i + 1], enc_zizi_fclayer2[i - 1]], axis=1)
                tmp = self.allzi_fc1[i](tmp)
                enc_allzi_fclayer2[i] = self.allzi_fc2[i](tmp)

                self.z_mean[i+1] = self.zi_mean[i](enc_allzi_fclayer2[i]) 
                self.z_logstd[i+1] = self.zi_logstd[i](enc_allzi_fclayer2[i]) 
                self.z_all[i+1] = self.sampling(self.z_mean[i+1], self.z_logstd[i+1], self.local_latent_dim)
                
        # returns latent feature
        return torch.cat([z_mean[i] for i in range(self.local_latent_num + 1)], axis=1)


class GenModel(nn.Module):
    def __init__(self, filename_obj, dim_in=512, centroid_scale=0.1, bias_scale=1.0, centroid_lr=0.1, bias_lr=1.0):
        super(GenModel, self).__init__()
        # load .obj
        self.template_mesh = sr.Mesh.from_obj(filename_obj)
        # vertices_base, faces = srf.load_obj(filename_obj)
        self.register_buffer('vertices_base', self.template_mesh.vertices.cpu()[0])#vertices_base)
        self.register_buffer('faces', self.template_mesh.faces.cpu()[0])#faces)

        self.nv = self.vertices_base.size(0)
        self.nf = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5

        dim = 1024
        dim_hidden = [dim, dim*2]
        self.fc1 = nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_centroid = nn.Linear(dim_hidden[1], 3)
        self.fc_bias = nn.Linear(dim_hidden[1], self.nv*3)
        
        self.dec_fc1 = nn.Linear(global_latent_dim+local_latent_num*local_latent_dim, 100 * (local_latent_num + 1))
        self.dec_fc2 = nn.Linear(100 * (local_latent_num + 1), 1024)
        self.dec_conv1 = nn.Deconv3D(kernel_size=4)
        self.dec_conv1 = nn.Deconv3D(kernel_size=5)

            'dec_conv2': tf.get_variable(name='dec_conv2', shape=[5, 5, 5, 32, 64],
                                          initializer=layers.xavier_initializer()),
            'dec_conv3': tf.get_variable(name='dec_conv3', shape=[6, 6, 6, 1, 32],
                                          initializer=layers.xavier_initializer()),
                                        
                                    
    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # decoder follows NMR
        centroid = self.fc_centroid(x) * self.centroid_scale

        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.nv, 3)

        base = self.vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

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

        self.inf_model = VSL_InfModel(im_size=args.image_size)
        self.gen_model = VSL_GenModel(filename_obj)
        self.image_decoder = VSL_ImageDecoder()
        self.renderer = sr.SoftRenderer(image_size=args.image_size, sigma_val=args.sigma_val, 
                                        aggr_func_rgb='hard', camera_mode='look_at', viewing_angle=15,
                                        dist_eps=1e-10)
        self.laplacian_loss = sr.LaplacianLoss(self.decoder.vertices_base, self.decoder.faces)
        self.flatten_loss = sr.FlattenLoss(self.decoder.faces)
        # TODO add latent code loss

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
        silhouettes = self.renderer(vertices, faces)
        return silhouettes.chunk(4, dim=0), laplacian_loss, flatten_loss

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
