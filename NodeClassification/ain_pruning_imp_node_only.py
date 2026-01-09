import numpy as np  
import torch  
import torch.nn as nn  
from net import net_gcn  
import pruning  
import utils  
import argparse  
import copy  
import warnings  
warnings.filterwarnings('ignore')  
  
def run_get_mask_node(args, imp_num, rewind_weight_mask, dataset_dict):  
    """带节点剪枝的掩码生成阶段"""  
    pruning.setup_seed(args.seed)  
      
    adj = dataset_dict['adj']  
    features = dataset_dict['features']  
    labels = dataset_dict['labels']  
    idx_train = dataset_dict['idx_train']  
    idx_val = dataset_dict['idx_val']  
    idx_test = dataset_dict['idx_test']  
      
    model = net_gcn(args.embedding_dim, adj)  
    pruning.add_mask(model)  
      
    # 添加节点掩码  
    pruning.add_node_mask(model)  
      
    if rewind_weight_mask is not None:
      # 1️ rewind GCN 权重
      model.load_state_dict(
          {k: v for k, v in rewind_weight_mask.items()
           if not k.startswith('node_mask')}
      )
  
      # 2️累计 hard node mask
      model.node_mask_fixed.data = rewind_weight_mask['node_mask_fixed'].clone()

    # 3️重新初始化 trainable mask
    model.node_mask_train.data.fill_(1.0) 
      
    pruning.add_trainable_mask_noise(model)  
    adj_spar, wei_spar = pruning.print_sparsity(model)  
      
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  
    model.cuda()  
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}  
    rewind_weight = copy.deepcopy(model.state_dict())  
      
    for epoch in range(1, args.total_epoch + 1):  
        model.train()  
        optimizer.zero_grad()  
          
        output = model(features, adj)  
        loss_train = nn.functional.nll_loss(output[idx_train], labels[idx_train])  
        loss_train.backward()  
          
        # 更新所有掩码的梯度  
        pruning.subgradient_update_mask_node(model, args)  
        optimizer.step()  
          
        with torch.no_grad():  
            model.eval()  
            output = model(features, adj)  
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])  
            acc_test = utils.accuracy(output[idx_test], labels[idx_test])  
              
            if acc_val > best_val_acc['val_acc']:  
                best_val_acc['test_acc'] = acc_test  
                best_val_acc['val_acc'] = acc_val  
                best_val_acc['epoch'] = epoch  
                rewind_weight, adj_spar, wei_spar, node_spar = pruning.get_final_mask_epoch_node(model, rewind_weight, args)  
          
        print("IMP[{}] (GCN {} Get Mask) Epoch:[{}/{}], Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}] | Adj:[{:.2f}%] Wei:[{:.2f}%] Node:[{:.2f}%]"  
             .format(imp_num, args.dataset, epoch, args.total_epoch, loss_train.item(),  
                    acc_val * 100, acc_test * 100, best_val_acc['val_acc'] * 100,  
                    best_val_acc['test_acc'] * 100, best_val_acc['epoch'], adj_spar, wei_spar, node_spar))  
      
    return rewind_weight  
  
def run_fix_mask_node(args, imp_num, rewind_weight_mask, dataset_dict):  
    """带节点剪枝的固定掩码训练阶段"""  
    pruning.setup_seed(args.seed)  
      
    adj = dataset_dict['adj']  
    features = dataset_dict['features']  
    labels = dataset_dict['labels']  
    idx_train = dataset_dict['idx_train']  
    idx_val = dataset_dict['idx_val']  
    idx_test = dataset_dict['idx_test']  
      
    model = net_gcn(args.embedding_dim, adj)  
    pruning.add_mask(model)  
    pruning.add_node_mask(model)  
    model.load_state_dict(rewind_weight_mask)  
      
    adj_spar, wei_spar = pruning.print_sparsity(model)  
    node_spar = pruning.print_node_sparsity(model)  
      
    # 固定所有掩码  
    for name, param in model.named_parameters():  
        if 'mask' in name:  
            param.requires_grad = False  
      
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  
    model.cuda()  
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0}  
      
    for epoch in range(1, args.total_epoch + 1):  
        model.train()  
        optimizer.zero_grad()  
          
        output = model(features, adj)  
        loss_train = nn.functional.nll_loss(output[idx_train], labels[idx_train])  
        loss_train.backward()  
        optimizer.step()  
          
        with torch.no_grad():  
            model.eval()  
            output = model(features, adj)  
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])  
            acc_test = utils.accuracy(output[idx_test], labels[idx_test])  
              
            if acc_val > best_val_acc['val_acc']:  
                best_val_acc['test_acc'] = acc_test  
                best_val_acc['val_acc'] = acc_val  
                best_val_acc['epoch'] = epoch  
          
        print("IMP[{}] (GCN {} Fix Mask) Epoch:[{}/{}], Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}] | Adj:[{:.2f}%] Wei:[{:.2f}%] Node:[{:.2f}%]"  
             .format(imp_num, args.dataset, epoch, args.total_epoch, loss_train.item(),  
                    acc_val * 100, acc_test * 100, best_val_acc['val_acc'] * 100,  
                    best_val_acc['test_acc'] * 100, best_val_acc['epoch'], adj_spar, wei_spar, node_spar))  
      
    print("Final: IMP[{}] (GCN {}) | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}] | Adj:[{:.2f}%] Wei:[{:.2f}%] Node:[{:.2f}%]"  
         .format(imp_num, args.dataset, best_val_acc['val_acc'] * 100,  
                best_val_acc['test_acc'] * 100, best_val_acc['epoch'], adj_spar, wei_spar, node_spar))  
  
def parser_loader():  
    parser = argparse.ArgumentParser(description='Options')  
    # 基础参数  
    parser.add_argument('--dataset', type=str, default='cora')  
    parser.add_argument('--embedding-dim', type=int, nargs='+', default=[1433, 512, 7])  
    parser.add_argument('--lr', type=float, default=0.008)  
    parser.add_argument('--weight-decay', type=float, default=8e-5)  
    parser.add_argument('--seed', type=int, default=1234)  
    parser.add_argument('--total_epoch', type=int, default=200)  
      
    # 边剪枝参数  
    parser.add_argument('--pruning_percent_adj', type=float, default=0.05)  
    parser.add_argument('--pruning_percent_wei', type=float, default=0.2)  
    parser.add_argument('--s1', type=float, default=1e-2)  
    parser.add_argument('--s2', type=float, default=1e-2)  
      
    # 节点剪枝参数  
    parser.add_argument('--pruning_percent_node', type=float, default=0.1)  
    parser.add_argument('--s3', type=float, default=1e-4)  
      
    # 其他参数  
    parser.add_argument('--init_soft_mask_type', type=str, default='all_one')  
      
    args = parser.parse_args()  
    return args  
  
if __name__ == "__main__":  
    args = parser_loader()  
    pruning.print_args(args)  
      
    # 加载数据  
    adj, features, labels, idx_train, idx_val, idx_test = utils.load_data(args.dataset)  
    features = torch.FloatTensor(features).cuda()  
    adj = torch.FloatTensor(adj).cuda()  
    labels = torch.LongTensor(labels).cuda()  
    idx_train = torch.LongTensor(idx_train).cuda()  
    idx_val = torch.LongTensor(idx_val).cuda()  
    idx_test = torch.LongTensor(idx_test).cuda()  
      
    dataset_dict = {  
        'adj': adj,  
        'features': features,  
        'labels': labels,  
        'idx_train': idx_train,  
        'idx_val': idx_val,  
        'idx_test': idx_test  
    }  
      
    rewind_weight = None  
      
    # 运行IMP迭代  
    for imp in range(1, 21):  
        print(f"\n{'='*50}")  
        print(f"IMP Iteration {imp}/20")  
        print(f"{'='*50}")  
          
        rewind_weight = run_get_mask_node(args, imp, rewind_weight, dataset_dict)  
        run_fix_mask_node(args, imp, rewind_weight, dataset_dict)
