import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np

from srcs.models.modnet import MODNet

from utils import load_modnet 

from dataloaders.dim_dataloader import DIMDataset

from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from srcs.models.utils import compute_mse, compute_sad
from tqdm import tqdm



if __name__ == '__main__':

    writer = SummaryWriter('./logs')

    # ------- 1. define loss function --------
    l1_loss = nn.L1Loss(reduce=False, size_average=False)
    l2_loss = nn.MSELoss(reduce=False, size_average=False)
    # alpha prediction loss: the abosolute difference between the ground truth alpha values and the
    # predicted alpha values at each pixel. However, due to the non-differentiable property of
    # absolute values, we use the following loss function to approximate it.
    def alpha_prediction_loss(y_pred, y_true):
        epsilon = 1e-6
        epsilon_sqr = epsilon ** 2
        mask = y_true[:, 0, :]
        diff = y_pred[:, 0, :] - y_true[:, 0, :]
        # diff = diff * mask
        num_pixels = torch.sum(mask)
        return torch.sum(torch.sqrt(torch.pow(diff, 2) + epsilon_sqr)) / (num_pixels + epsilon)
    
    #l2_loss
    def get_sementic_loss(pred_semantic, pred_semantic_mixed=None, y_a=None,  y_b=None, 
                          mixed_loss=True, base_weight=1, gamma=1):
        batch_size = pred_semantic.shape[0]
        features = pred_semantic.view(batch_size, -1)
        features_mixed = pred_semantic_mixed.view(batch_size, -1)

        features = features / features.norm(dim=1, keepdim=True)
        features_mixed = features_mixed / features_mixed.norm(dim=1, keepdim=True)
        features_logits = features @ features_mixed.t()
        
        modulating_factor = torch.softmax(features_logits, dim=-1)
        #features_pt = torch.softmax(features_logits, dim=-1)
        features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(pred_semantic.device)
        #step 2: supervised learning loss
        modulating_factor = modulating_factor.gather(1, features_ground_truth).detach().clone()
        
        epsilon = 1e-6
        epsilon_sqr = epsilon ** 2
        w = (base_weight+modulating_factor)**gamma

        if mixed_loss:
            loss = (pred_semantic_mixed-y_b)**2
        else:
            loss = (pred_semantic-y_a)**2
        loss = torch.mean(loss, dim=-1)
        #loss = w*loss
            
        return 0.5*loss.mean()
    
    #l1_loss
    def get_detail_loss(pred_detail, pred_detail_mixed=None, y_a=None, y_b=None, md_masks=None, md_masks_mixed=None,
                          mixed_loss=True, base_weight=1, gamma=1):
        batch_size = pred_detail.shape[0]
        features = pred_detail.view(batch_size, -1)
        features_mixed = pred_detail_mixed.view(batch_size, -1)

        features = features / features.norm(dim=1, keepdim=True)
        features_mixed = features_mixed / features_mixed.norm(dim=1, keepdim=True)
        features_logits = features @ features_mixed.t()
        
        modulating_factor = torch.softmax(features_logits, dim=-1)
        #features_pt = torch.softmax(features_logits, dim=-1)
        features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(pred_semantic.device)
        #step 2: supervised learning loss
        modulating_factor = modulating_factor.gather(1, features_ground_truth).detach().clone()
        
        epsilon = 1e-6
        epsilon_sqr = epsilon ** 2
        w = (base_weight+modulating_factor)**gamma

        if mixed_loss:
            loss = torch.sum(md_masks_mixed * torch.abs(pred_detail_mixed.view(batch_size, y_b.shape[1], -1)-y_b), dim=-1).view(batch_size, -1)
            #loss = w*loss
            loss = loss.sum()/ md_masks_mixed.sum()

        else:
            #x = torch.abs(pred_detail.view(batch_size, y_a.shape[1], -1)-y_a)
            #print("x.shape=", x.shape)
            loss = torch.sum(md_masks * torch.abs(pred_detail.view(batch_size, y_a.shape[1], -1)-y_a), dim=-1).view(batch_size, -1)
            #loss = w*loss
            loss = loss.sum()/ md_masks.sum()

            
        return loss

    def get_fusion_loss(pred_matte, pred_matte_mixed=None, y_a=None, y_b=None,
                          mixed_loss=True, base_weight=1, gamma=1):
        batch_size = pred_matte.shape[0]
        features = pred_matte.view(batch_size, -1)
        features_mixed = pred_matte_mixed.view(batch_size, -1)

        features = features / features.norm(dim=1, keepdim=True)
        features_mixed = features_mixed / features_mixed.norm(dim=1, keepdim=True)
        features_logits = features @ features_mixed.t()
        
        modulating_factor = torch.softmax(features_logits, dim=-1)
        #features_pt = torch.softmax(features_logits, dim=-1)
        features_ground_truth = torch.arange(batch_size, dtype=torch.long).view(-1, 1).to(pred_semantic.device)
        #step 2: supervised learning loss
        modulating_factor = modulating_factor.gather(1, features_ground_truth).detach().clone()
        
        epsilon = 1e-6
        epsilon_sqr = epsilon ** 2
        w = (base_weight+modulating_factor)**gamma
        h, w = pred_matte.shape[-2], pred_matte.shape[-1]

        if mixed_loss:
            loss = torch.abs(pred_matte_mixed.view(batch_size, y_b.shape[1], -1)-y_b)
            #loss = torch.abs(pred_matte.view(batch_size, y_a.shape[1], -1)-y_a)
            #print("loss1.shape=", loss.shape)
            loss = torch.sum(loss, dim=-1)
            #print("loss2.shape=", loss.shape)
            loss = torch.sum(loss, dim=-1, keepdim=True)
            loss = loss / (h*w)

            
            #loss = w*loss
            loss = loss.mean()

        else:
            #x = torch.abs(pred_detail.view(batch_size, y_a.shape[1], -1)-y_a)
            #print("x.shape=", x.shape)
            
            loss = torch.abs(pred_matte.view(batch_size, y_a.shape[1], -1)-y_a)
            #print("loss1.shape=", loss.shape)
            loss = torch.sum(loss, dim=-1)
            #print("loss2.shape=", loss.shape)
            loss = torch.sum(loss, dim=-1, keepdim=True)
            loss = loss / (h*w)
            
            #loss = w*loss
            loss = loss.mean()

            
        return loss
    
    def evaluate(net, test_loader):
        net.eval()
        with torch.no_grad():
            pred_all, labels_all, trimap_all = [], [], []
            for data in tqdm(test_loader):
                inputs, labels, trimap = data[0], data[1], data[4]
                h, w = inputs.shape[-2], inputs.shape[-1]
                batch_size =inputs.shape[0]
                #print('inputs.shape=', inputs.shape, "   labels.shape=", labels.shape, '   down_labels.shape=', down_labels.shape)
                #print("md_masks.shape=", md_masks.shape)
                inputs = inputs.float().cuda()
                labels = labels.float()
                
                _, _, pred_matte = net(inputs, False)
                
                labels_all += [labels.view(batch_size, h, w).numpy()]
                trimap_all += [trimap.view(batch_size, h, w).numpy()]
                pred_all += [pred_matte.view(batch_size, h, w).cpu().numpy()]
                
            labels_all = np.concatenate(labels_all, 0)
            trimap_all = np.concatenate(trimap_all, 0)
            pred_all = np.concatenate(pred_all, 0)
            mse = compute_mse(pred_all, labels_all, trimap_all)
            sad = compute_sad(pred_all, labels_all)
        #print("mse=", mse, "  sad=", sad)
        
        return mse, sad



    def mixup_data(x, y, y_down, md_masks, alpha=1.0):
        if alpha>0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        
        mixed_x = lam * x + (1-lam)*x[index, :]
        y_b = lam*y + (1-lam)*y[index]
        y_down_mixed = lam*y_down + (1-lam)*y_down[index]
        md_masks_mixed = lam*md_masks + (1-lam)*md_masks[index]
        
        return mixed_x, y_b, y_down_mixed, md_masks_mixed, lam




    # ------- 2. set the directory of training dataset --------
    epoch_num = 40
    bs = 16
    train_num = 0
    val_num = 0
    momentum = 0.9
    steps = [10, 20, 30] #[40, 80, 120]
    weight_decay = 0.0005
    lr = 0.01
    gamma = 0.1


    alpha = 0.2
    mixed_loss=False
    base_weight = 0.5
    upper_gamma = 1
    
    resume = False
    resume_ckpt = './pretrained/modnet_portrait.ckpt'
    pre_train_dir = "./pretrained/modnet_pretrained.ckpt"

    log_save_path = "./logs/dim/"
    pre_train_dir = "./pretrained/modnet_pretrained.ckpt"
    
    #training set
    fg_train_dir = 'data/DIM/Combined_Dataset/Training_set/Adobe-licensed images/fg/'
    matte_train_dir = 'data/DIM/Combined_Dataset/Training_set/Adobe-licensed images/alpha/'
    bg_train_dir = 'data/DIM/train2014/'
    
    #testing set
    fg_test_dir = 'data/DIM/Combined_Dataset/Test_set/Adobe-licensed images/fg/'
    matte_test_dir = 'data/DIM/Combined_Dataset/Test_set/Adobe-licensed images/alpha/'
    bg_test_dir = 'data/DIM/VOC2008/'

    
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
    
    train_dataset = DIMDataset(
                        split='train', 
                        fg_dir=fg_train_dir, 
                        matte_dir=matte_train_dir, 
                        bg_dir=bg_train_dir)
                            
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)

    test_dataset = DIMDataset(
                        split='valid', 
                        fg_dir=fg_test_dir, 
                        matte_dir=matte_test_dir, 
                        bg_dir=bg_test_dir)
                            
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4)

    # ------- 3. define model --------
    # define the net
    net = MODNet(backbone_pretrained=True)
    #net = load_modnet(pre_train_dir)
    if torch.cuda.is_available():
        net =  torch.nn.DataParallel(net).cuda()


    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=steps, gamma=gamma)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_sem_loss = 0.0
    running_det_loss = 0.0
    running_fus_loss = 0.0
    ite_num4val = 0
    print_frq = 500
    save_frq = 1 # save the model every 2000 iterations


    for epoch in range(0, epoch_num):
        net.train()
        #sad, mse = evaluate(net, test_loader)

        for data in tqdm(train_loader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1
            inputs, labels, down_labels, md_masks = data[0], data[1], data[2], data[3]
            #print('inputs.shape=', inputs.shape, "   labels.shape=", labels.shape, '   down_labels.shape=', down_labels.shape)
            #print("md_masks.shape=", md_masks.shape)
            inputs = inputs.float().cuda()
            labels = labels.float().cuda()
            md_masks = md_masks.float().cuda()
            down_labels = down_labels.float().cuda()

            inputs_mixed, labels_mixed, down_labels_mixed, md_masks_mixed, lam = mixup_data(inputs, labels, down_labels, md_masks, alpha=alpha)

            # forward + backward + optimize
            pred_semantic, pred_detail, pred_matte = net(inputs, False)
            pred_semantic_mixed, pred_detail_mixed, pred_matte_mixed = net(inputs_mixed, False)
            pred_semantic, pred_detail, pred_matte = pred_semantic.squeeze(1), pred_detail.squeeze(1), pred_matte.squeeze(1)
            pred_semantic_mixed, pred_detail_mixed, pred_matte_mixed  = pred_semantic_mixed.squeeze(1), pred_detail_mixed.squeeze(1), pred_matte_mixed.squeeze(1) 
            #print("shape=shape=", pred_matte.shape, "   pred_semantic.shape=", pred_semantic.shape, "  pred_detail.shape=", pred_detail.shape)
            detail_loss1 = (md_masks * l1_loss(pred_detail, labels)).sum() / md_masks.sum() #no compositional loss
            #detail_loss1 = (md_masks * l1_loss(pred_detail, labels)).sum() 
            
            #detail_loss1 = l1_loss(pred_detail, labels) #no compositional loss
            
            detail_loss = get_detail_loss(pred_detail, pred_detail_mixed=pred_detail_mixed, 
                                        y_a=labels,  y_b=labels_mixed, 
                                        md_masks=md_masks,
                                        md_masks_mixed=md_masks_mixed,
                                        mixed_loss=mixed_loss, 
                                        base_weight=base_weight, gamma=upper_gamma)
            
            
            semantic_loss1 = 0.5 * l2_loss(pred_semantic, down_labels).mean()
            #semantic_loss1 = l2_loss(pred_semantic, down_labels)
            #print("semtic1.shape=", semantic_loss1)
            semantic_loss = get_sementic_loss(pred_semantic, pred_semantic_mixed=pred_semantic_mixed, 
                                        y_a=down_labels,  y_b=down_labels_mixed, 
                                        mixed_loss=mixed_loss, 
                                        base_weight=base_weight, gamma=upper_gamma)
            
            #print("semantic1=", semantic_loss1, "   \n semantic=", semantic_loss)

            Lc = alpha_prediction_loss(pred_matte, labels)
            #print("Lc=", Lc)
            #fusion_loss1 = l1_loss(pred_matte, labels).mean() + Lc
            #fusion_loss1 = l1_loss(pred_matte, labels).mean()
            #print("fusion1.shape=", fusion_loss1.shape)
            fusion_loss = get_fusion_loss(pred_matte, pred_matte_mixed=pred_matte_mixed, 
                                        y_a=labels, y_b=labels_mixed,
                                        mixed_loss=mixed_loss, 
                                        base_weight=base_weight, 
                                        gamma=upper_gamma)
            fusion_loss = fusion_loss + Lc
            
            loss = semantic_loss + 10 * detail_loss + fusion_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_det_loss += detail_loss.item()
            running_fus_loss += fusion_loss.item()
            running_sem_loss += semantic_loss.item()

            if ite_num % 10==0:
                print("[epoch: %3d /%3d, ite: %d],  lr: %5f, det_loss: %3f fus_loss: %3f sem_loss: %3f train_loss: %3f" % (
                    epoch + 1, epoch_num, ite_num, \
                    scheduler.get_last_lr()[0], \
                    running_det_loss / ite_num4val, \
                    running_fus_loss / ite_num4val, \
                    running_sem_loss / ite_num4val, \
                    running_loss / ite_num4val, \
                    ))

            

        
        #torch.save(net.state_dict(), model_dir + '/' + model_name + "_epoch_%d_train_%3f_tar_%3f.pth" % (
        #    epoch, running_loss / ite_num4val, running_loss / ite_num4val))
        checkpoint_name = "train_%d.pth" %(epoch+1)
        torch.save(net.state_dict(), log_save_path + '/' + checkpoint_name)
        

        sad, mse = evaluate(net, test_loader)

        print("[epoch: %3d /%3d, ite: %d],  lr: %5f, det_loss: %3f fus_loss: %3f sem_loss: %3f train_loss: %3f   mse:%6f   sad:%6f" % (
                epoch + 1, epoch_num, ite_num, \
                scheduler.get_last_lr()[0], \
                running_det_loss / ite_num4val, \
                running_fus_loss / ite_num4val, \
                running_sem_loss / ite_num4val, \
                running_loss / ite_num4val, \
                mse, sad))

        writer.add_scalar('lr', scheduler.get_last_lr()[0], ite_num / print_frq)
        writer.add_scalar('total loss', running_loss / ite_num4val, ite_num / print_frq)
        writer.add_scalar('semantic loss', running_sem_loss / ite_num4val, ite_num / print_frq)
        writer.add_scalar('detail loss', running_det_loss / ite_num4val, ite_num / print_frq)

        writer.add_image('semantic image', make_grid([pred_semantic[0], down_labels[0]]), ite_num / print_frq)
        writer.add_image('detail image', make_grid([pred_detail[0], md_masks[0], md_masks[0] * labels[0]]), ite_num / print_frq)
        writer.add_image('fusion image', make_grid([pred_matte[0], labels[0]]), ite_num / print_frq)
        writer.add_image('original image', make_grid(inputs[0]), ite_num / print_frq)

        running_loss = 0.0
        running_det_loss = 0.0
        running_sem_loss = 0.0
        running_fus_loss = 0.0
        ite_num4val = 0

        scheduler.step()
