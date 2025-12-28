import json  
import matplotlib.pyplot as plt  
import numpy as np  
  
def plot_sparsity_accuracy(results_file):  
    with open(results_file, 'r') as f:  
        results = json.load(f)  
      
    adj_sparsities = [r['adj_spar'] for r in results]  
    wei_sparsities = [r['wei_spar'] for r in results]  
    test_accs = [r['final_test'] for r in results]  
      
    plt.figure(figsize=(12, 5))  
      
    # 邻接矩阵稀疏度 vs 精度  
    plt.subplot(1, 2, 1)  
    plt.plot(adj_sparsities, test_accs, 'bo-', label='Adjacency Pruning')  
    plt.xlabel('Adjacency Sparsity (%)')  
    plt.ylabel('Test Accuracy (%)')  
    plt.title('Adjacency Sparsity vs Accuracy')  
    plt.grid(True)  
    plt.legend()  
      
    # 权重稀疏度 vs 精度  
    plt.subplot(1, 2, 2)  
    plt.plot(wei_sparsities, test_accs, 'ro-', label='Weight Pruning')  
    plt.xlabel('Weight Sparsity (%)')  
    plt.ylabel('Test Accuracy (%)')  
    plt.title('Weight Sparsity vs Accuracy')  
    plt.grid(True)  
    plt.legend()  
      
    plt.tight_layout()  
    plt.savefig('sparsity_accuracy_curves.png')  
    plt.show()  
  
# 使用示例  
plot_sparsity_accuracy('imp_all_results.json')


def plot_mask_distribution(model, epoch, acc_test, path):  
    print("Plot Epoch:[{}] Test Acc[{:.2f}]".format(epoch, acc_test * 100))  
    if not os.path.exists(path): os.makedirs(path)  
    adj_mask, weight_mask = get_mask_distribution(model)  
  
    plt.figure(figsize=(15, 5))  
    plt.subplot(1,2,1)  
    plt.hist(adj_mask)  
    plt.title("adj mask")  
    plt.xlabel('mask value')  
    plt.ylabel('times')  
  
    plt.subplot(1,2,2)  
    plt.hist(weight_mask)  
    plt.title("weight mask")  
    plt.xlabel('mask value')  
    plt.ylabel('times')  
    plt.suptitle("Epoch:[{}] Test Acc[{:.2f}]".format(epoch, acc_test * 100))  
    plt.savefig(path + '/mask_epoch{}.png'.format(epoch))

def plot_comprehensive_comparison(results_file):  
    with open(results_file, 'r') as f:  
        results = json.load(f)  
      
    imp_nums = list(range(1, len(results) + 1))  
    adj_sparsities = [r['adj_spar'] for r in results]  
    wei_sparsities = [r['wei_spar'] for r in results]  
    test_accs = [r['final_test'] * 100 for r in results]  
    val_accs = [r['highest_valid'] * 100 for r in results]  
      
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))  
      
    # 稀疏度变化  
    ax1.plot(imp_nums, adj_sparsities, 'b-', label='Adjacency Sparsity')  
    ax1.plot(imp_nums, wei_sparsities, 'r-', label='Weight Sparsity')  
    ax1.set_xlabel('IMP Iteration')  
    ax1.set_ylabel('Sparsity (%)')  
    ax1.set_title('Sparsity Progression')  
    ax1.legend()  
    ax1.grid(True)  
      
    # 精度变化  
    ax2.plot(imp_nums, test_accs, 'g-', label='Test Accuracy')  
    ax2.plot(imp_nums, val_accs, 'orange', label='Best Validation Accuracy')  
    ax2.set_xlabel('IMP Iteration')  
    ax2.set_ylabel('Accuracy (%)')  
    ax2.set_title('Accuracy Progression')  
    ax2.legend()  
    ax2.grid(True)  
      
    # 稀疏度-精度散点图  
    scatter = ax3.scatter(adj_sparsities, test_accs, c=imp_nums, cmap='viridis', s=50)  
    ax3.set_xlabel('Adjacency Sparsity (%)')  
    ax3.set_ylabel('Test Accuracy (%)')  
    ax3.set_title('Sparsity vs Accuracy (colored by iteration)')  
    plt.colorbar(scatter, ax=ax3, label='IMP Iteration')  
    ax3.grid(True)  
      
    plt.tight_layout()  
    plt.savefig('comprehensive_analysis.png')  
    plt.show()
