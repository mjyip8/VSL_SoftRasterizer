import torch
import torch.nn as nn
import torch.nn.functional as F

import soft_renderer as sr
import soft_renderer.functional as srf
import math

class VSLModel(nn.Module):
    def __init__(self, args):
        super(VSLModel, self).__init__()
        self.batch_size = args.batch_size
        self.inf_model = InfModel(args=args)
        self.gen_model = GenModel(args=args)
        self.image_decoder = ImageDecoder(args=args)
        self.renderer = sr.SoftRenderer(image_size=args.image_size, sigma_val=args.sigma_val, 
                                        aggr_func_rgb='hard', camera_mode='look_at', viewing_angle=15,
                                        dist_eps=1e-10)
        self.laplacian_loss = sr.LaplacianLoss(self.gen_model.vertices_base, self.gen_model.faces)
        self.flatten_loss = sr.FlattenLoss(self.gen_model.faces)
        # TODO add latent code loss

    def model_param(self):
        return list(self.inf_model.parameters()) + \
                list(self.gen_model.parameters()) + \
                list(self.image_decoder.parameters())

    def set_sigma(self, sigma):
        self.renderer.set_sigma(sigma)

    def reconstruct(self, images, voxels = None):
        latent_feature = None 
        learned_feature = None
        vertices = None 
        faces = None
        if (voxels != None):
            latent_feature = self.inf_model(voxels)
            vertices, faces = self.gen_model(latent_feature)
        else:
            learned_feature = self.image_decoder(images)
            vertices, faces = self.gen_model(learned_feature)

            # learned_feature = self.image_decoder(images)
            # vertices, faces = self.gen_model(learned_feature)

        return vertices, faces, latent_feature, learned_feature

    def predict_multiview(self, image_a, image_b, viewpoint_a, viewpoint_b, voxels, i):
        # [Ia, Ib]
        images = torch.cat((image_a, image_b), dim=0)

        # [Va, Va, Vb, Vb], set viewpoints
        viewpoints = torch.cat((viewpoint_a, viewpoint_a, viewpoint_b, viewpoint_b), dim=0)
        self.renderer.transform.set_eyes(viewpoints)

        vertices, faces, latent_feature, learned_feature = self.reconstruct(images, voxels)

        laplacian_loss = self.laplacian_loss(vertices)
        flatten_loss = self.flatten_loss(vertices)

        # Compute weight for latent loss
        gamma = torch.tensor([[5e-3]]).cuda()
        t = torch.tensor([[float(i)]]).cuda()
        if (i <= 50):
            gamma = torch.pow(10, torch.floor(t / 10.) - 8.)
        elif (50 < t < 100):
            gamma = torch.floor((t - 40.) / 10.) * 0.001

        latent_loss = 0.
        if (latent_feature != None and learned_feature != None):
            latent_loss = gamma/2. * torch.norm(latent_feature - learned_feature[:self.batch_size]) 
            latent_loss += (gamma/2. * torch.norm(latent_feature - learned_feature[self.batch_size:]) )

        # [Ma, Mb, Ma, Mb]
        vertices = torch.cat((vertices, vertices, vertices, vertices), dim=0)
        faces = torch.cat((faces, faces, faces, faces), dim=0)

        # [Raa, Rba, Rab, Rbb], cross render multiview images
        silhouettes = self.renderer(vertices, faces)
        return silhouettes.chunk(4, dim=0), laplacian_loss, flatten_loss, latent_loss

    def evaluate_iou(self, images, voxels):
        vertices, faces, _ , _ = self.reconstruct(images)

        faces_ = srf.face_vertices(vertices, faces).data
        faces_norm = faces_ * 1. * (32. - 1) / 32. + 0.5
        voxels_predict = srf.voxelization(faces_norm, 32, False).cpu().numpy()
        voxels_predict = voxels_predict.transpose(0, 2, 1, 3)[:, :, :, ::-1]
        iou = (voxels * voxels_predict).sum((1, 2, 3)) / (0 < (voxels + voxels_predict)).sum((1, 2, 3))
        return iou, vertices, faces

    def forward(self, images=None, viewpoints=None, voxels=None, task='train', i = 0):
        if task == 'train':
            return self.predict_multiview(images[0], images[1], viewpoints[0], viewpoints[1], voxels, i)
        elif task == 'test':
            return self.evaluate_iou(images, voxels)


class InfModel(nn.Module):
    def __init__(self, args):
        super(InfModel, self).__init__()
        self.im_size = args.image_size
        self.batch_size = args.batch_size
        self.global_latent_dim = args.global_latent_dim
        self.local_latent_dim = args.local_latent_dim
        self.local_latent_num = args.local_latent_num
        dim_in=1
        dim1=32
        dim_hidden = [dim1, dim1*2, dim1*4, 256, 100]

        self.conv1 = nn.ModuleList([nn.Conv3d(dim_in, dim_hidden[0], kernel_size=5, stride=2, padding=0) for _ in range(self.local_latent_num+1)])
        self.conv2 = nn.ModuleList([nn.Conv3d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=0) for _ in range(self.local_latent_num+1)])
        self.conv3 = nn.ModuleList([nn.Conv3d(dim_hidden[1], dim_hidden[2], kernel_size=5, stride=2, padding=0) for _ in range(self.local_latent_num+1)])

        self.fc1 = nn.ModuleList([nn.Linear(128, dim_hidden[3]) for _ in range(self.local_latent_num+1)])
        self.fc2 = nn.ModuleList([nn.Linear(dim_hidden[3], dim_hidden[4]) for _ in range(self.local_latent_num+1)])

        self.z_mean_fc = nn.Linear(dim_hidden[4], self.global_latent_dim)
        self.z_logstd_fc = nn.Linear(dim_hidden[4], self.global_latent_dim)

        self.zzi_fc1 = nn.ModuleList([nn.Linear(self.global_latent_dim, dim_hidden[4]) for _ in range(self.local_latent_num)])
        self.zzi_fc2 = nn.ModuleList([nn.Linear(dim_hidden[4], dim_hidden[4]) for _ in range(self.local_latent_num)])

        self.bn1 = nn.BatchNorm3d(dim_hidden[0])
        self.bn2 = nn.BatchNorm3d(dim_hidden[1])
        self.bn3 = nn.BatchNorm3d(dim_hidden[2])

        self.zizi_fc1 = nn.ModuleList([nn.Linear(self.local_latent_dim, dim_hidden[4]) for _ in range(self.local_latent_num)])
        self.zizi_fc2 = nn.ModuleList([nn.Linear(dim_hidden[4], dim_hidden[4]) for _ in range(self.local_latent_num)])

        self.allzi_fc1 = nn.ModuleList([nn.Linear(200, 100)] + [nn.Linear(300, 100) for _ in range(self.local_latent_num-1)])
        self.allzi_fc2 = nn.ModuleList([nn.Linear(100, 100) for _ in range(self.local_latent_num)])
        self.zi_mean = nn.ModuleList([nn.Linear(100, self.local_latent_dim) for _ in range(self.local_latent_num)])
        self.zi_logstd = nn.ModuleList([nn.Linear(100, self.local_latent_dim) for _ in range(self.local_latent_num)])

    def sampling(self, z_mean, z_logstd, latent_dim):
        epsilon = torch.randn((self.batch_size, latent_dim)).cuda()
        return z_mean + torch.exp(z_logstd) * epsilon


    def forward(self, x):
        enc_fc2, z_mean, z_logstd, z_all, k1_loss = ([torch.zeros((self.batch_size, 100)).cuda() for _ in range(self.local_latent_num + 1)] for _ in range(5))
        # # input_shape x -> local_lat z_i
        for i in range(self.local_latent_num + 1):
            tmp = self.conv1[i](x)
            tmp = F.relu(self.bn1(tmp), inplace=True)
            tmp = F.relu(self.bn2(self.conv2[i](tmp)), inplace=True)
            tmp = F.relu(self.bn3(self.conv3[i](tmp)), inplace=True)
            tmp = torch.reshape(tmp, (self.batch_size, -1))
            tmp = F.relu(self.fc1[i](tmp), inplace=True)
            enc_fc2[i] = F.relu(self.fc2[i](tmp), inplace=True)
        
        # # sample global latent variable
        z_mean[0] = self.z_mean_fc(enc_fc2[0])
        z_logstd[0] = self.z_logstd_fc(enc_fc2[0])
        z_all[0] = self.sampling(z_mean[0], z_logstd[0], self.global_latent_dim)
        
        enc_zzi_fclayer2, enc_allzi_fclayer2 = ([torch.zeros((100, self.local_latent_num)).cuda() for _ in range(self.local_latent_num)] for _ in range(2))
        enc_zizi_fclayer2 = [torch.zeros((self.im_size, self.local_latent_num - 1)).cuda() for _ in range(self.local_latent_num - 1)]

        for i in range(self.local_latent_num):
            # z -> z_i
            tmp = F.relu(self.zzi_fc1[i](z_all[0]))
            enc_zzi_fclayer2[i] = F.relu(self.zzi_fc2[i](tmp))
            
            if i == 0:  # sampling z_1
                tmp = torch.cat([enc_zzi_fclayer2[i], enc_fc2[i+1]], axis=1)
                tmp = self.allzi_fc1[i](tmp) 
                enc_allzi_fclayer2[i] = self.allzi_fc2[i](tmp) 

                z_mean[1] = self.zi_mean[i](enc_allzi_fclayer2[i]) 
                z_logstd[1] = self.zi_logstd[i](enc_allzi_fclayer2[i]) 
                z_all[1] = self.sampling(z_mean[1], z_logstd[1], self.local_latent_dim) 
            else:   # sampling z_i (i >= 1)   
                tmp = F.relu(self.zizi_fc1[i-1](z_all[i]))
                enc_zizi_fclayer2[i - 1] = F.relu(tmp)
             
                tmp = torch.cat([enc_zzi_fclayer2[i], enc_fc2[i + 1], enc_zizi_fclayer2[i - 1]], axis=1)
                tmp = self.allzi_fc1[i](tmp)
                enc_allzi_fclayer2[i] = self.allzi_fc2[i](tmp)

                z_mean[i+1] = self.zi_mean[i](enc_allzi_fclayer2[i]) 
                z_logstd[i+1] = self.zi_logstd[i](enc_allzi_fclayer2[i]) 
                z_all[i+1] = self.sampling(z_mean[i+1], z_logstd[i+1], self.local_latent_dim)
        return torch.cat([z_mean[i] for i in range(self.local_latent_num + 1)], axis=1)

class GenModel(nn.Module):
    def __init__(self, args):
        super(GenModel, self).__init__()

        self.filename_obj = args.filename_obj
        self.batch_size = args.batch_size
        self.global_latent_dim = args.global_latent_dim
        self.local_latent_dim = args.local_latent_dim
        self.local_latent_num = args.local_latent_num
        # load .obj
        self.template_mesh = sr.Mesh.from_obj(self.filename_obj)
        # vertices_base, faces = srf.load_obj(filename_obj)
        self.register_buffer('vertices_base', self.template_mesh.vertices.cpu()[0])#vertices_base)
        self.register_buffer('faces', self.template_mesh.faces.cpu()[0])#faces)

        dim_in=512
        centroid_scale=0.1
        bias_scale=1.0
        centroid_lr=0.1

        self.nv = self.vertices_base.size(0)
        self.nf = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5

        dim = 1024
        dim_hidden = [dim, dim*2]
        self.fc1 = nn.Linear(self.global_latent_dim+self.local_latent_num*self.local_latent_dim, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_centroid = nn.Linear(dim_hidden[1], 3)
        self.fc_bias = nn.Linear(dim_hidden[1], self.nv*3)
        
                                    
    def forward(self, x):
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
        faces = self.faces[None, :, :].repeat(self.batch_size, 1, 1)

        return vertices, faces

class ImageDecoder(nn.Module):
    def __init__(self, args):
        super(ImageDecoder, self).__init__()
        dims_in = 4
        dim_hidden = [16, 32, 64, 128]
        self.batch_size = args.batch_size
        self.p_drop = args.p_drop
        self.image_conv1 = nn.Conv2d(dims_in, dim_hidden[0], kernel_size=16, stride=2, padding=0)
        self.image_conv2 = nn.Conv2d(dim_hidden[0], dim_hidden[1], kernel_size=5, stride=2, padding=0)
        self.image_conv3 = nn.Conv2d(dim_hidden[1], dim_hidden[2], kernel_size=2, stride=2, padding=0)
        self.image_conv4 = nn.Conv2d(dim_hidden[2], dim_hidden[3], kernel_size=2, stride=2, padding=0)     
        self.image_fclayer1 = nn.Linear(512, 200)
        self.image_fcdropout = nn.Dropout(p=self.p_drop)
        self.image_fclayer2 = nn.Linear(200, args.global_latent_dim + args.local_latent_num * args.local_latent_dim)


    def forward(self, y):
        y = F.relu(self.image_conv1(y))
        y = F.relu(self.image_conv2(y))
        y = F.relu(self.image_conv3(y))
        y = F.relu(self.image_conv4(y))
        y = torch.reshape(y, (self.batch_size * 2, 512))
        y = F.relu(self.image_fclayer1(y))
        y = self.image_fcdropout(y)
        y = F.relu(self.image_fclayer2(y))
        return y
