import torch
import h5py, numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PhaseClassifier import logging
logger = logging.getLogger(__name__)

def observable_from_sim(path):
    ###from simulation
    with h5py.File(path, 'r') as file:
        f = {}
        logger.info("Keys: %s" % list(file.keys()))
        for key in file.keys():
            f[key] = torch.tensor(np.array(file[key]))

    f['Temperature'].unique()
    mag_sim = []
    susc = []
    tmp = []
    for i,temp in enumerate(f['Temperature'].unique()):
        # Get the indices of the samples with the current temperature
        indices = (f['Temperature'] == temp).nonzero(as_tuple=True)[0]
        # Get the corresponding predictions
        mag_sim.append(f['Magnetization'][indices,0].abs().mean())
        susc.append(f['Susceptibility'][indices,0].abs().mean())
        tmp.append(temp)
    return mag_sim, susc, tmp

def plot_magnetization(mag, tmp, mag_sim):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.errorbar(mag[:,0], mag[:,1], yerr=mag[:,2].sqrt(), fmt='o-', lw=2.5, label="Classifier")
    ax.errorbar(mag[:,0], mag[:,3], yerr = mag[:,4].sqrt(), fmt='o-', label="Magnetization")
    ax.fill_between(mag[:,0], mag[:,1]-mag[:,2].sqrt(), mag[:,1]+mag[:,2].sqrt(), color='lightblue', alpha=0.5, label='Classifier std')
    ax.plot(tmp, mag_sim, 'o-', lw=2.5, label="Simulated Magnetization")

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean Prediction")
    ax.set_title("Mean Prediction vs Temperature")
    ax.legend()
    ax.hlines([0.5], xmin=2.20, xmax=2.3, colors='r', linestyles='dashed')
    ax.vlines([2.269], ymin=0, ymax=1, colors='r', linestyles='dashed')
    return fig

def plot_susc(mag, tmp, susc, num_of_spins=128*128):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(mag[:,0], num_of_spins*(mag[:,2])/mag[:,0], 'o-', lw=2.5, label="Classifier Variance")
    # ax.plot(mag[:,0], 32*32*(mag[:,4])/mag[:,0], 'o-', lw=2.5, label="Magnetization Variance")
    ax.plot(tmp, susc, 'o-', lw=2.5, label="Simulated Susceptibility")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean Prediction")
    ax.set_title("Mean Prediction vs Temperature")
    ax.legend()
    ax.vlines([2.269], ymin=-0.3, ymax=0, colors='r', linestyles='dashed')
    return fig

def generate_data_to_plot(self):
    # collect all labels from the test loader into a single 1D tensor (on CPU)
    self.model.eval()
    with torch.no_grad():
        pred_list = []
        temp_list = []
        mag_list = []
        for loader in [self.data_mgr_test.train_loader,self.data_mgr_test.val_loader,self.data_mgr_test.test_loader]:
            for (x,y,t) in loader:
                x = x.to(self.device, dtype=torch.float64)
                pred = self.model(x)
                # pred_list.append(F.softmax(pred).argmax(dim=1).cpu())
                pred_list.append(F.softmax(pred)[:,1].cpu())
                temp_list.append(t[:,0].cpu())
                mag_list.append(x.mean(dim=(1,2,3)).cpu())
        test_preds = torch.cat(pred_list, dim=0)
        test_temps = torch.cat(temp_list, dim=0)
        test_mags = torch.cat(mag_list, dim=0)

    mag = torch.zeros(test_temps.unique().shape[0],5)
    for i,temp in enumerate(test_temps.unique()):
        # Get the indices of the samples with the current temperature
        indices = (test_temps == temp).nonzero(as_tuple=True)[0]
        # Get the corresponding predictions
        preds = test_preds[indices]
        mag[i,:] = torch.tensor([temp, preds.mean(), preds.var(), test_mags[indices].abs().mean(),test_mags[indices].abs().var()])
        # Print some basic statistics
        # print(f"Temperature: {temp.item()}, Num samples: {len(indices)}, Predicted phases: {preds.mean()}")

    return mag

def cm(self):
    self.model.eval()
    with torch.no_grad():
        pred_list = []
        target_list = []
        for loader in [self.data_mgr.train_loader,self.data_mgr.val_loader,self.data_mgr.test_loader]:
            for (x,y,t) in loader:
                x = x.to(self.device, dtype=torch.float64)
                pred = self.model(x)
                pred_list.append(F.softmax(pred).argmax(dim=1).cpu())
                # pred_list.append(F.softmax(pred)[:,1].cpu())
                target_list.append(y[:,0].cpu())
        test_preds = torch.cat(pred_list, dim=0)
        test_targets = torch.cat(target_list, dim=0)

    # Compute confusion matrix: [[TN, FP], [FN, TP]]
    tn = int(((test_preds == 0) & (test_targets == 0)).sum().item())
    fp = int(((test_preds == 1) & (test_targets == 0)).sum().item())
    fn = int(((test_preds == 0) & (test_targets == 1)).sum().item())
    tp = int(((test_preds == 1) & (test_targets == 1)).sum().item())
    cm = torch.tensor([[tn, fp], [fn, tp]])

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm.numpy(), cmap='Blues', vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{(cm/cm.sum(dim=1).unsqueeze(1))[i,j].item():.2f}", ha='center', va='center', color='black', fontsize=14)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks([0,1]); ax.set_xticklabels(['0','1'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['0','1'])
    ax.set_title('Confusion Matrix')
    plt.colorbar(im, ax=ax)
    plt.show()