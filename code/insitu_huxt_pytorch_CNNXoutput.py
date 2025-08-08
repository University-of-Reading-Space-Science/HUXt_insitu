# -*- coding: utf-8 -*-
"""
A script ot train the CNN for solar wind backmapping

@author: mathe
"""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import joblib
import huxt as H
import huxt_inputs as Hin
import huxt_analysis as HA
import astropy.units as u
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import onnxruntime as ort

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


rmin = 21.5*u.solRad
rmax = 215*u.solRad


cwd = os.path.dirname(os.path.realpath(__file__))
figdir =  os.path.join(os.path.dirname(cwd), 'figures' )
data_dir = os.path.join(os.path.dirname(cwd), 'data' )

download_now = False #flag to download the MAS data. Extracted profiles are provided, so not required.




# <codecell> download and process required MAS solutions

if download_now:
    crstart = 1861
    crstop = 2296
    
    Ncr = crstop-crstart
    Nphi = 128
    
    dlats = [-10,-5,0,5,10]
    Nlat = len(dlats)
    
    X = np.empty((Ncr*Nlat, Nphi)) 
    Y = np.empty((Ncr*Nlat, Nphi)) 
    
    
    
    counter = 0
    
    for cr in range(crstart, crstop):
    
        for lat  in dlats:
            v_boundary = Hin.get_MAS_long_profile(cr, lat*u.deg)
            #v_boundary = np.ones((128)) * 400 *u.km/u.s
            
            #reduce the speeds to 21.5 rS, but don't apply a longitude shift - it changes the structure and phase doesn't matter
            v_boundary_21p5 = v_boundary.copy()
            for i, v in enumerate(v_boundary):
                v_boundary_21p5[i], phi = Hin.map_v_inwards(v, 30*u.solRad, 0*u.rad, 21.5*u.solRad)
            
            # Setup HUXt to do a 0  day simulation
            model = H.HUXt(v_boundary=v_boundary_21p5,  simtime=0.1*u.day, dt_scale=4,
                            r_min=rmin, r_max=rmax)
            
            # Solve these conditions, with no ConeCMEs added.
            cme_list = []
            model.solve(cme_list)
            
            #get the values at Earth
            v_inner_lon = model.v_grid[0,0,:]
            v_Earth_lon = model.v_grid[0,-1,:]
            
     
            
            #now map ballistically back to the inner boundary
            #================================================
            #Map from 1-au rto 21.5 rS
            vcarr_rmin = Hin.map_v_boundary_inwards(v_Earth_lon, 215*u.solRad, 21.5*u.solRad)
    
           
            prior = vcarr_rmin
            truth = v_inner_lon
            #save the data
            X[counter, :] = prior
            Y[counter,:] = truth
            
            counter = counter+1
            
    
    # save the data
    
    np.savetxt(os.path.join(data_dir, 'X.txt'), X, delimiter=",")
    np.savetxt(os.path.join(data_dir, 'Y.txt'), Y, delimiter=",")

# <codecell> Train model with pytorch


# Load data
X = np.loadtxt(os.path.join(data_dir, 'X.txt'), delimiter=",")
Y = np.loadtxt(os.path.join(data_dir, 'Y.txt'), delimiter=",")
n_lons = X.shape[1]

# Preprocess & Reshape for Conv1D
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)
Y_scaled = y_scaler.fit_transform(Y)

# Reshape for Conv1D: (samples, length, channels)
X_scaled = X_scaled[..., np.newaxis]
Y_scaled = Y_scaled[..., np.newaxis]

# Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)  # shape: (batch, channels, length)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).permute(0, 2, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).permute(0, 2, 1)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the CNN model
class Conv1DCNN(nn.Module):
    def __init__(self, input_length):
        super(Conv1DCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.model(x)

model = Conv1DCNN(n_lons)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), os.path.join(data_dir, 'CNN_torch_weights.pth'))


# Export model to ONNX
onnx_path = os.path.join(data_dir, 'CNN_model.onnx')

# Create a dummy input with same shape as one batch (batch_size=1, channels=1, length=n_lons)
dummy_input = torch.randn(1, 1, n_lons).to(device)

# Export the model
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11  # OR 17, depending on your ONNX version
)

print(f"ONNX model saved to: {onnx_path}")


# Save the scalers
joblib.dump(x_scaler, os.path.join(data_dir, 'x_scaler_torch.save'))
joblib.dump(y_scaler, os.path.join(data_dir, 'y_scaler_torch.save'))



# <codecell>  load the model back in and apply it to the test inputs
 
# Load scalers
x_scaler = joblib.load(os.path.join(data_dir, 'x_scaler_torch.save'))
y_scaler = joblib.load(os.path.join(data_dir, 'y_scaler_torch.save'))

Y_test_np = Y_test.squeeze()  # (batch, length)
X_test_np = X_test.squeeze()  # (batch, length)

# Inverse transform the original test data
Y_true = y_scaler.inverse_transform(Y_test_np)
Y_backmap = x_scaler.inverse_transform(X_test_np)



y_scaler = joblib.load(os.path.join(data_dir, 'y_scaler_torch.save'))

# Load test data (X_test) from earlier session or rerun the preprocessing
# Assuming X_test is still available in memory
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to("cpu")  # (batch, channels, length)

# Initialize and load model
#model = Conv1DCNN(X_test_tensor.shape[2])
# Define the model architecture again
class Conv1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Create model instance and load weights
cnn = Conv1DCNN()
cnn.load_state_dict(torch.load(os.path.join(data_dir, 'CNN_torch_weights.pth')))
cnn.eval()


# Predict
with torch.no_grad():
    Y_pred_tensor = cnn(X_test_tensor)  # (batch, channels, length)

# Convert predictions and targets back to numpy
Y_pred_scaled = Y_pred_tensor.permute(0, 2, 1).squeeze().numpy()

# Inverse transform
Y_pred = y_scaler.inverse_transform(Y_pred_scaled)



# <codecell> compute the MAE at 21.5 rS over the test data


#compute MAE over non-training data
mae_bm = []
mae_nn = []
for idx in range(0, len(X_test)):
    # Predict one sample
    #idx = 1
    
    mae_bm.append(np.mean(abs(Y_true[idx,:] - Y_backmap[idx,:])) )
    mae_nn.append(np.mean(abs(Y_true[idx,:] - Y_pred[idx])) )
    

plt.figure()
plt.hist(mae_bm, label = 'MAE: backmap')
plt.hist(mae_nn, label = 'MAE: backmap + CNN')
plt.legend()


# <codecell>  ONNX: load the model back in and apply it to the test inputs
 
# Load scalers
x_scaler = joblib.load(os.path.join(data_dir, 'x_scaler_torch.save'))
y_scaler = joblib.load(os.path.join(data_dir, 'y_scaler_torch.save'))

Y_test_np = Y_test.squeeze()  # (batch, length)
X_test_np = X_test.squeeze()  # (batch, length)

# Inverse transform the original test data
Y_true = y_scaler.inverse_transform(Y_test_np)
Y_backmap = x_scaler.inverse_transform(X_test_np)



y_scaler = joblib.load(os.path.join(data_dir, 'y_scaler_torch.save'))

# Load test data (X_test) from earlier session or rerun the preprocessing
# Assuming X_test is still available in memory
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to("cpu")  # (batch, channels, length)



# Load the ONNX model
onnx_path = os.path.join(data_dir, 'CNN_model.onnx')
ort_session = ort.InferenceSession(onnx_path)

# Prepare input: convert to numpy, shape (batch, channels, length)
X_input = np.transpose(X_test, (0, 2, 1)).astype(np.float32)  # X_test shape: (batch, length, 1)

# Run inference
input_name = ort_session.get_inputs()[0].name
outputs = ort_session.run(None, {input_name: X_input})
Y_pred_onnx = outputs[0]  # shape: (batch, 1, length)

# Postprocess
Y_pred_scaled = np.transpose(Y_pred_onnx, (0, 2, 1)).squeeze()  # shape: (batch, length)
Y_pred = y_scaler.inverse_transform(Y_pred_scaled)



#compute MAE over non-training data
mae_bm = []
mae_nn = []
for idx in range(0, len(X_test)):
    # Predict one sample
    #idx = 1
    
    mae_bm.append(np.mean(abs(Y_true[idx,:] - Y_backmap[idx,:])) )
    mae_nn.append(np.mean(abs(Y_true[idx,:] - Y_pred[idx])) )
    

plt.figure()
plt.hist(mae_bm, label = 'MAE: backmap')
plt.hist(mae_nn, label = 'MAE: backmap + CNN')
plt.legend()


# <codecell> # Plot Example Result
dlon = 360/len(Y_true[0,:])
carr_lons = np.arange(dlon/2, 360, dlon)

i = np.argmax(mae_bm)
#i = 50  # Pick a sample index

#run HUXt out with each of these
model_true = H.HUXt(v_boundary=Y_true[i]*u.km/u.s,  simtime=0.1*u.day, dt_scale=4,
               r_min=rmin, r_max = 215*u.solRad)
model_true.solve([])
model_backmap = H.HUXt(v_boundary=Y_backmap[i]*u.km/u.s,  simtime=0.1*u.day, dt_scale=4,
               r_min=rmin, r_max = 215*u.solRad)
model_backmap.solve([])
model_cnn = H.HUXt(v_boundary=Y_pred[i]*u.km/u.s,  simtime=0.1*u.day, dt_scale=4,
               r_min=rmin , r_max = 215*u.solRad)
model_cnn.solve([])

nr = np.argmin(model_true.v_grid[0,:,0].value - 0.3*215 )

fig = plt.figure( figsize=(12, 10))
gs = gridspec.GridSpec(4, 3, hspace=0.1, wspace=0.1)

ax1 = fig.add_subplot(gs[0, 0], projection='polar')         
ax2 = fig.add_subplot(gs[0, 1], projection='polar')         
ax3 = fig.add_subplot(gs[0, 2], projection='polar')  

ax4 = fig.add_subplot(gs[1, :])  
ax5 = fig.add_subplot(gs[2, :])  
ax6 = fig.add_subplot(gs[3, :])  
       

mae_bm = np.nanmean(abs( Y_true[i] - Y_backmap[i]))
mae_cnn = np.nanmean(abs( Y_true[i] - Y_pred[i]))
ax4.plot(carr_lons, Y_true[i], 'k', label="True")
ax4.plot(carr_lons, Y_backmap[i], 'b', label="Backmap")
ax4.plot(carr_lons, Y_pred[i], 'r--', label="Backmap+CNN")
ax4.set_ylabel(r'$V_{SW}$ [km/s]')
#axes[0].set_ylabel(r'$V_{SW}$ [km/s]')
#axes[0].title("CNN Output Example")
ax4.set_ylim((250, 750))
ax4.set_xticks((0, 90, 180, 270, 360 ))
ax4.set_xlim((0, 360))
ax4.text(0.02, 0.95, '(a) 0.1 AU', transform=ax4.transAxes,
        fontsize=14, verticalalignment='top', horizontalalignment='left')
ax4.text(0.18, 0.95, 'MAE = ' + format(mae_bm, ".1f") + r' km s$^{-1}$', transform=ax4.transAxes,
        fontsize=14, verticalalignment='top', horizontalalignment='left', color = 'b')
ax4.text(0.43, 0.95, 'MAE = ' + format(mae_cnn, ".1f") + r' km s$^{-1}$', transform=ax4.transAxes,
        fontsize=14, verticalalignment='top', horizontalalignment='left', color = 'r')
ax4.set_xticklabels([])



mae_bm = np.nanmean(abs(model_true.v_grid[0,nr,:] - model_backmap.v_grid[0,nr,:])).value
mae_cnn = np.nanmean(abs(model_true.v_grid[0,nr,:] - model_cnn.v_grid[0,nr,:])).value
ax5.plot(carr_lons, model_true.v_grid[0,nr,:], 'k', label="True")
ax5.plot(carr_lons, model_backmap.v_grid[0,nr,:], 'b', label="Backmap")
ax5.plot(carr_lons, model_cnn.v_grid[0,nr,:], 'r--', label="Backmap+CNN")
ax5.set_ylabel(r'$V_{SW}$ [km/s]')
#axes[0].title("CNN Output Example")
#axes[1].legend(ncol =3)
ax5.set_ylim((250, 750))
ax5.set_xticks((0, 90, 180, 270, 360 ))
ax5.set_xlim((0, 360))
ax5.text(0.02, 0.95, '(b) 0.3 AU', transform=ax5.transAxes,
        fontsize=14, verticalalignment='top', horizontalalignment='left')
ax5.text(0.18, 0.95, 'MAE = ' + format(mae_bm, ".1f") + r' km s$^{-1}$', transform=ax5.transAxes,
        fontsize=14, verticalalignment='top', horizontalalignment='left', color = 'b')
ax5.text(0.43, 0.95, 'MAE = ' + format(mae_cnn, ".1f") + r' km s$^{-1}$', transform=ax5.transAxes,
        fontsize=14, verticalalignment='top', horizontalalignment='left', color = 'r')
ax5.set_xticklabels([])


mae_bm = np.nanmean(abs(model_true.v_grid[0,-1,:] - model_backmap.v_grid[0,-1,:])).value
mae_cnn = np.nanmean(abs(model_true.v_grid[0,-1,:] - model_cnn.v_grid[0,-1,:])).value
ax6.plot(carr_lons, model_true.v_grid[0,-1,:], 'k', label="True")
ax6.plot(carr_lons, model_backmap.v_grid[0,-1,:], 'b', label="Backmap")
ax6.plot(carr_lons, model_cnn.v_grid[0,-1,:], 'r--', label="Backmap+CNN")
ax6.set_ylabel(r'$V_{SW}$ [km/s]')
ax6.set_xlabel(r'Carrington longitude [deg]')
#axes[0].title("CNN Output Example")
#axes[1].legend(ncol =3)
ax6.set_ylim((250, 750))
ax6.set_xticks((0, 90, 180, 270, 360 ))
ax6.set_xlim((0, 360))
ax6.text(0.02, 0.95, '(c) 1 AU', transform=ax6.transAxes,
        fontsize=14, verticalalignment='top', horizontalalignment='left')
ax6.text(0.18, 0.95, 'MAE = ' + format(mae_bm, ".1f") + r' km s$^{-1}$', transform=ax6.transAxes,
        fontsize=14, verticalalignment='top', horizontalalignment='left', color = 'b')
ax6.text(0.43, 0.95, 'MAE = ' + format(mae_cnn, ".1f") + r' km s$^{-1}$', transform=ax6.transAxes,
        fontsize=14, verticalalignment='top', horizontalalignment='left', color = 'r')


HA.plot(model_true, 0*u.day, fighandle = fig, axhandle= ax1, minimalplot=True)
HA.plot(model_backmap, 0*u.day, fighandle = fig, axhandle= ax2, minimalplot=True)
HA.plot(model_cnn, 0*u.day, fighandle = fig, axhandle= ax3, minimalplot=True)


#add a colourbar
cax = fig.add_axes([0.92, 0.70, 0.02, 0.15])  # Adjust as needed
# Define normalization and colormap
norm = mcolors.Normalize(vmin=200, vmax=810)
cmap = cm.viridis  # or any other colormap

# Create a ScalarMappable and add colorbar
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # Needed to avoid warnings

cbar = fig.colorbar(sm, cax=cax)
fig.text(0.90, 0.87, 'V [km/s]', va='center', rotation='horizontal',
         fontsize =16)

#add the legend
ax4.legend(ncol =3, loc = 'lower center', bbox_to_anchor=(0.51, 2.15), columnspacing=7.5)

fig.savefig(os.path.join(figdir, 'backmap_example.pdf'))

# <codecell> run HUXt to compute MAE at 1 au over the test data

i_0p3au = np.argmin(abs(model_true.r.value - 0.3*215) )
i_1au = np.argmin(abs(model_true.r.value - 1*215) )






mae_1au_bm =[]
mae_1au_nn = []

mae_inner_bm =[]
mae_inner_nn = []

mae_0p3au_bm =[]
mae_0p3au_nn = []

for vin_bm, vin_cnn, vin_true in zip(Y_backmap, Y_pred, Y_true):
    # Setup HUXt to do a 5 day simulation, with model output every 4 timesteps (roughly half and hour time step), looking at 0 longitude
    model = H.HUXt(v_boundary=vin_true*u.km/u.s, lon_out=0.0*u.deg, simtime=27.27*u.day, dt_scale=4,
                   r_min=rmin)
    
    # Solve these conditions, with no ConeCMEs added.
    model.solve([])
    ts_true_inner = model.v_grid[:,0,0]
    ts_true_1au = model.v_grid[:,i_1au,0]
    ts_true_0p3au = model.v_grid[:,i_0p3au,0]

    #Earth_ts_true = HA.get_observer_timeseries(model, observer='Earth', suppress_warning = True)
    
    
    # Setup HUXt to do a 5 day simulation, with model output every 4 timesteps (roughly half and hour time step), looking at 0 longitude
    model = H.HUXt(v_boundary=vin_bm*u.km/u.s, lon_out=0.0*u.deg, simtime=27.27*u.day, dt_scale=4,
                   r_min=rmin)
    
    # Solve these conditions, with no ConeCMEs added.
    model.solve([])
    ts_bm_inner = model.v_grid[:,0,0]
    ts_bm_1au = model.v_grid[:,i_1au,0]
    ts_bm_0p3au = model.v_grid[:,i_0p3au,0]
    
    
    # Setup HUXt to do a 5 day simulation, with model output every 4 timesteps (roughly half and hour time step), looking at 0 longitude
    model = H.HUXt(v_boundary=vin_cnn*u.km/u.s, lon_out=0.0*u.deg, simtime=27.27*u.day, dt_scale=4,
                   r_min=rmin)
    
    # Solve these conditions, with no ConeCMEs added.
    model.solve([])
    ts_cnn_inner = model.v_grid[:,0,0]
    ts_cnn_1au = model.v_grid[:,i_1au,0]
    ts_cnn_0p3au = model.v_grid[:,i_0p3au,0]
    
    mae_1au_bm.append(np.mean(abs(ts_bm_1au - ts_true_1au)) )
    mae_1au_nn.append(np.mean(abs(ts_cnn_1au - ts_true_1au)) )
    
    mae_inner_bm.append(np.mean(abs(ts_bm_inner - ts_true_inner)) )
    mae_inner_nn.append(np.mean(abs(ts_cnn_inner - ts_true_inner)) )
    
    mae_0p3au_bm.append(np.mean(abs(ts_bm_0p3au - ts_true_0p3au)) )
    mae_0p3au_nn.append(np.mean(abs(ts_cnn_0p3au - ts_true_0p3au)) )
    

# <codecell> plot the histograms of MAE

yy = (0,0.32)
xx = (0, 40)



fig, axes = plt.subplots(1, 3, figsize=(15, 4))


bm = np.array([q.value for q in mae_inner_bm])
nn = np.array([q.value for q in mae_inner_nn])

axes[0].hist(bm, color = 'b', density=True, label = 'Backmap', alpha=0.6)
axes[0].hist(nn, color = 'r',density=True, label = 'Backmap + CNN', alpha=0.6)
axes[0].set_xlabel(r'$V_R$ MAE @ 0.1 AU [km/s]')
axes[0].set_ylabel(r'Prob. density')
#axes[0].set_ylim(yy)
axes[0].set_xlim(xx)
#axes[0].legend()
axes[0].legend(bbox_to_anchor=(2.4, 1.25), ncol=2)
axes[0].text(0.01, 0.98, "(a)",  transform=axes[0].transAxes,  ha='left', va='top', fontsize = 14,) 
axes[0].text(0.98, 0.98, "<MAE> = " + format(np.nanmean(bm), ".1f"),  transform=axes[0].transAxes,  
                                             ha='right', va='top', fontsize = 14, color = 'b') 
axes[0].text(0.98, 0.88, "<MAE> = " + format(np.nanmean(nn), ".1f"),  transform=axes[0].transAxes,  
                                             ha='right', va='top', fontsize = 14, color = 'r') 

bm = np.array([q.value for q in mae_0p3au_bm])
nn = np.array([q.value for q in mae_0p3au_nn])
axes[1].hist(bm, color = 'b', density=True, label = 'Backmap', alpha=0.6)
axes[1].hist(nn, color = 'r',density=True, label = 'Backmap + CNN', alpha=0.6)
axes[1].set_xlabel(r'$V_R$ MAE @ 0.3 AU [km/s]')
#axes[1].set_ylim(yy)
axes[1].set_xlim(xx)
#axes[1].legend()
axes[1].text(0.01, 0.98, "(b)",  transform=axes[1].transAxes,  ha='left', va='top', fontsize = 14,) 
axes[1].text(0.98, 0.98, "<MAE> = " + format(np.nanmean(bm), ".1f"),  transform=axes[1].transAxes,  
                                             ha='right', va='top', fontsize = 14, color = 'b') 
axes[1].text(0.98, 0.88, "<MAE> = " + format(np.nanmean(nn), ".1f"),  transform=axes[1].transAxes,  
                                             ha='right', va='top', fontsize = 14, color = 'r') 

bm = np.array([q.value for q in mae_1au_bm])
nn = np.array([q.value for q in mae_1au_nn])
axes[2].hist(bm, color = 'b', density=True, label = 'Backmap', alpha=0.6)
axes[2].hist(nn, color = 'r',density=True, label = 'Backmap + CNN', alpha=0.6)
axes[2].set_xlabel(r'$V_R$ MAE @ 1 AU [km/s]')
#axes[2].set_ylim(yy)
axes[2].set_xlim(xx)
axes[2].text(0.01, 0.98, "(c)",  transform=axes[2].transAxes,  ha='left', va='top', fontsize = 14,) 
axes[2].text(0.98, 0.98, "<MAE> = " + format(np.nanmean(bm), ".1f"),  transform=axes[2].transAxes,  
                                             ha='right', va='top', fontsize = 14, color = 'b') 
axes[2].text(0.98, 0.88, "<MAE> = " + format(np.nanmean(nn), ".1f"),  transform=axes[2].transAxes,  
                                             ha='right', va='top', fontsize = 14, color = 'r') 

fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.8, hspace=0.1)
#plt.tight_layout()
fig.savefig(os.path.join(figdir, 'MAE_cnn.pdf'))

