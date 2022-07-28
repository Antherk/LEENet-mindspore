import os
import mindspore as ms
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 
import argparse
from decom import DecomNet
from ajust import AjustNet
from tools import weights_init
from dataset import leenet_loader
from mindvision.engine.callback import LossMonitor

import mindspore.nn as nn
import mindspore.dataset as ds

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_context(device_target='GPU')
    ms.set_context(device_id=0)
    # -------------------------------------Define Network and Init----------------------------
    '''
    Network_create-> Network_init -> optimizer_init -> pretrained_loading
    '''
    Decomnet = DecomNet()
    for m in Decomnet.cells_and_names():
        weights_init(m[1])
    
    Decomnet_optimizer = nn.Adam(params=Decomnet.get_parameters(), learning_rate=config.lr)
    if config.load_Decomnet_pretrain == True:
        Decomnet.load_state_dict(ms.load_checkpoint(config.pretrain_Decomnet_dir))
    
    Ajustnet = AjustNet().cuda()
    for m in Ajustnet.cells_and_names():
        weights_init(m[1])
    Ajustnet_optimizer = nn.Adam(params=Decomnet.get_parameters(), learning_rate=config.lr)
    if config.load_Ajustnet_pretrain == True:
        Decomnet.load_state_dict(ms.load_checkpoint(config.pretrain_Ajustnet_dir))
    '''
    # multi GPUS
    Decomnet = torch.nn.DataParallel(Dec, device_ids=range(torch.cuda.device_count()))
    Ajustnet = torch.nn.DataParallel(Enh, device_ids=range(torch.cuda.device_count()))
    '''

    # -------------------------------------Dataset preparing------------------------------
    
    train_dataset = leenet_loader(config.images_path)
    train_loader = ds.GeneratorDataset(train_dataset,
                                        shuffle=True,
                                        num_parallel_workers=config.num_workers)
    train_loader._batch_size = config.train_batch_size

 # -------------------------------------Training loop------------------------------------------
    loss_ajust = 1
    loss_decom = 1
    model_decom = ms.Model(Decomnet, loss_fn=loss_decom, optimizer=Decomnet_optimizer)
    model_ajust = ms.Model(Ajustnet, loss_fn=loss_ajust, optimizer=Ajustnet_optimizer)
    model_decom.train(config.num_epochs, train_loader, callbacks=[LossMonitor(config.lr)])

    '''
    Decomnet.train()
    Ajustnet.train()
    for epoch in range(config.num_epochs):
        # ---------------------------------train one epoch------------------------------
        for iteration, (image, label) in enumerate(train_loader):
            image = image.unsqueeze(0).cuda()
            label = label.unsqueeze(0).cuda()
            BRDF, F = Decomnet(image)
            F_delta = Ajustnet()
            # ---------------------------------Define Loss Function-------------------------------------
            L_decom_BRDF  = 1
            L_decom_recon = 1
            loss_decom = 1
            loss_ajust =1
            # loss_reconst_dec = reconst_loss(S_low, R_low.mul(I_low)) \
            #             + reconst_loss(S_normal, R_normal.mul(I_normal)) \
            #             + 0.001 * reconst_loss(S_low, R_normal.mul(I_low)) \
            #             + 0.001 * reconst_loss(S_normal, R_low.mul(I_normal))
            # loss_ivref = 0.01 * reconst_loss(R_low, R_normal)
            # loss_dec = loss_reconst_dec + loss_ivref
            # smooth_loss_low = get_gradients_loss(I_low, R_low)
            # smooth_loss_normal = get_gradients_loss(I_normal, R_normal)
            # smooth_loss_low_hat = get_gradients_loss(I_low_hat, R_low)
            # loss_dec += 0.1 * smooth_loss_low + 0.1 * smooth_loss_normal
            
            # ---------------------------------optimer network per iter-------------------------------------
            Decomnet_optimizer.zero_grad()
            loss_decom.backward()
            torch.nn.utils.clip_grad_norm(Decomnet.parameters(),config.grad_clip_norm)
            Decomnet_optimizer.step()

            Ajustnet_optimizer.zero_grad()
            loss_ajust.backward()
            torch.nn.utils.clip_grad_norm(Ajustnet.parameters(),config.grad_clip_norm)
            Ajustnet_optimizer.step()
    '''


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1) #梯度裁剪
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    
    parser.add_argument('--images_path', type=str, default="1/")
    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--save_iter', type=int, default=10)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots_Zero_DCE++/")
    
    #pretrained
    parser.add_argument('--load_Decomnet_pretrain', type=bool, default= False)
    parser.add_argument('--pretrain_Decomnet_dir', type=str, default= "snapshots_Zero_DCE++/Epoch99.pth")
    parser.add_argument('--load_Ajustnet_pretrain', type=bool, default= False)
    parser.add_argument('--pretrain_Ajustnet_dir', type=str, default= "snapshots_Zero_DCE++/Epoch99.pth")
    
    config = parser.parse_args()
    
    train(config)