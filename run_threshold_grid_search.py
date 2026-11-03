import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
import csv

BASE_PATH='Goddess_Split'
VALIDATION_PATH=os.path.join(BASE_PATH,'validation')
TEST_PATH=os.path.join(BASE_PATH,'test')
OOD_DIR='New_Test_Images'
MODEL_PATH=os.path.join(BASE_PATH,'goddess_material_resnet18_v3_best.pth')
IMAGE_SIZE=224

GODDESSES=["Annapurna","Durga","Kali","Lakshmi","Lalitha","Saraswati"]
MATERIALS=["Brass","Bronze","Copper","Silver","Stone","Wood"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',device)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# model
model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Identity()
model.goddess_classifier = nn.Linear(num_features, len(GODDESSES))
model.material_classifier = nn.Linear(num_features, len(MATERIALS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

# helper dataset
class DM(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.samples=[]
        for r,dirs,files in os.walk(root):
            for f in files:
                if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
                p=os.path.join(r,f); rp=os.path.relpath(p,root); parts=rp.split(os.sep)
                g=None;m=None
                for name in GODDESSES:
                    if name in parts: g=name; break
                for name in MATERIALS:
                    if name in parts: m=name; break
                if g and m: self.samples.append((p,GODDESSES.index(g),MATERIALS.index(m)))
        print('Loaded',len(self.samples),'from',root)
        self.transform=transform
    def __len__(self): return len(self.samples)
    def __getitem__(self,i): p,g,m=self.samples[i]; img=Image.open(p).convert('RGB'); return self.transform(img), g, m, p

val_ds = DM(VALIDATION_PATH, transform)

# compute validation logits, features, and max_sim
val_logits_g=[]
val_logits_m=[]
val_feats=[]
val_true_g=[]
val_true_m=[]
val_paths=[]
for i in range(len(val_ds)):
    x,g_true,m_true,p = val_ds[i]
    with torch.no_grad():
        xt=x.unsqueeze(0).to(device)
        feat = model(xt).cpu().numpy().reshape(-1)
        g_logit = model.goddess_classifier(torch.tensor(feat).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
        m_logit = model.material_classifier(torch.tensor(feat).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
    val_logits_g.append(g_logit)
    val_logits_m.append(m_logit)
    val_feats.append(feat)
    val_true_g.append(g_true)
    val_true_m.append(m_true)
    val_paths.append(p)
val_logits_g=np.stack(val_logits_g,axis=0)
val_logits_m=np.stack(val_logits_m,axis=0)
val_feats=np.stack(val_feats,axis=0)
val_true_g=np.array(val_true_g)
val_true_m=np.array(val_true_m)

# compute per-material centroids
centroids=[]
for mi in range(len(MATERIALS)):
    feats_m = val_feats[val_true_m==mi]
    if len(feats_m)==0: centroids.append(np.zeros(val_feats.shape[1]))
    else: centroids.append(np.mean(feats_m,axis=0))
centroids=np.stack(centroids,axis=0)
centroids_norm = centroids / np.clip(np.linalg.norm(centroids,axis=1,keepdims=True),1e-8,None)

# compute max_sim per validation image
val_max_sim = []
for f in val_feats:
    fn = f.reshape(1,-1); fn = fn/(np.linalg.norm(fn)+1e-8)
    sims = cosine_similarity(fn, centroids_norm)[0]
    val_max_sim.append(float(np.max(sims)))
val_max_sim = np.array(val_max_sim)

# Temperatures from previous calibration
best_g_T = 1.0
best_m_T = 1.2

# compute calibrated confidences
val_probs_g = softmax(val_logits_g / best_g_T, axis=1)
val_probs_m = softmax(val_logits_m / best_m_T, axis=1)
val_conf_g = np.max(val_probs_g, axis=1)
val_conf_m = np.max(val_probs_m, axis=1)
val_pred_g = np.argmax(val_probs_g, axis=1)
val_pred_m = np.argmax(val_probs_m, axis=1)

# grid search
reco_thrs = np.arange(0.30, 0.96, 0.05)
cent_thrs = np.arange(0.60, 0.91, 0.05)
results=[]
for r in reco_thrs:
    for c in cent_thrs:
        accepted_mask = (val_conf_g >= r) & (val_conf_m >= r) & (val_max_sim >= c)
        accepted_idxs = np.where(accepted_mask)[0]
        acc_count = len(accepted_idxs)
        if acc_count>0:
            material_acc = (val_pred_m[accepted_idxs] == val_true_m[accepted_idxs]).sum() / acc_count
            goddess_acc = (val_pred_g[accepted_idxs] == val_true_g[accepted_idxs]).sum() / acc_count
        else:
            material_acc = None
            goddess_acc = None
        results.append({'reco_thr':float(r),'centroid_thr':float(c),'accepted_count':int(acc_count),'material_acc': material_acc,'goddess_acc':goddess_acc})

# choose best by material_acc (max), tie-breaker: larger accepted_count
valid_results = [r for r in results if r['material_acc'] is not None]
if not valid_results:
    raise SystemExit('No thresholds produced any accepted validation images')
# sort by material_acc desc, accepted_count desc
sorted_results = sorted(valid_results, key=lambda x: (x['material_acc'], x['accepted_count']), reverse=True)
best = sorted_results[0]

# write grid csv
os.makedirs('debug_outputs', exist_ok=True)
with open('debug_outputs/threshold_grid_search.csv','w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f, fieldnames=['reco_thr','centroid_thr','accepted_count','material_acc','goddess_acc'])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print('Best thresholds found (maximize material acc on accepted validation images):')
print(best)

# Now evaluate best thresholds on TEST and OOD
# prepare test set
# load test images and compute logits, feats

# helper to process a folder of labeled images similarly
from glob import glob

def load_dataset(root):
    paths=[]; true_g=[]; true_m=[]
    for r,dirs,files in os.walk(root):
        for f in files:
            if not f.lower().endswith(('.jpg','.jpeg','.png')): continue
            p=os.path.join(r,f); rp=os.path.relpath(p,root); parts=rp.split(os.sep)
            g=None; m=None
            for name in GODDESSES:
                if name in parts: g=name; break
            for name in MATERIALS:
                if name in parts: m=name; break
            if g and m:
                paths.append(p); true_g.append(GODDESSES.index(g)); true_m.append(MATERIALS.index(m))
    return paths, np.array(true_g), np.array(true_m)

# test
test_paths, test_true_g, test_true_m = load_dataset(TEST_PATH)
print('Test size',len(test_paths))
# compute
test_pred_m = []
test_pred_g = []
test_conf_m = []
test_conf_g = []
test_max_sim = []
for p in test_paths:
    img = Image.open(p).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).cpu().numpy().reshape(-1)
        g_logit = model.goddess_classifier(torch.tensor(feat).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
        m_logit = model.material_classifier(torch.tensor(feat).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
    g_probs = softmax(g_logit / best_g_T)
    m_probs = softmax(m_logit / best_m_T)
    g_conf = float(np.max(g_probs)); m_conf=float(np.max(m_probs))
    g_pred = int(np.argmax(g_probs)); m_pred=int(np.argmax(m_probs))
    ms = float(np.max(cosine_similarity(feat.reshape(1,-1)/(np.linalg.norm(feat)+1e-8), centroids_norm)[0]))
    test_pred_m.append(m_pred); test_pred_g.append(g_pred); test_conf_m.append(m_conf); test_conf_g.append(g_conf); test_max_sim.append(ms)

# apply best thresholds
r_thr = best['reco_thr']; c_thr = best['centroid_thr']
accepted_mask_test = (np.array(test_conf_g) >= r_thr) & (np.array(test_conf_m) >= r_thr) & (np.array(test_max_sim) >= c_thr)
accepted_idxs = np.where(accepted_mask_test)[0]
accepted_count = len(accepted_idxs)
rejected_count = len(test_paths)-accepted_count
material_acc_on_accepted = (np.array(test_pred_m)[accepted_idxs] == test_true_m[accepted_idxs]).sum() / accepted_count if accepted_count>0 else None
goddess_acc_on_accepted = (np.array(test_pred_g)[accepted_idxs] == test_true_g[accepted_idxs]).sum() / accepted_count if accepted_count>0 else None

# OOD
ood_paths = [os.path.join(OOD_DIR,f) for f in sorted(os.listdir(OOD_DIR)) if f.lower().endswith(('.jpg','.jpeg','.png'))] if os.path.exists(OOD_DIR) else []
ood_total = len(ood_paths)
ood_accepted = 0
ood_accepted_list = []
for p in ood_paths:
    img=Image.open(p).convert('RGB')
    x=transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(x).cpu().numpy().reshape(-1)
        g_logit = model.goddess_classifier(torch.tensor(feat).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
        m_logit = model.material_classifier(torch.tensor(feat).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
    g_conf = float(np.max(softmax(g_logit / best_g_T)))
    m_conf = float(np.max(softmax(m_logit / best_m_T)))
    ms = float(np.max(cosine_similarity(feat.reshape(1,-1)/(np.linalg.norm(feat)+1e-8), centroids_norm)[0]))
    acc = (g_conf >= r_thr) and (m_conf >= r_thr) and (ms >= c_thr)
    if acc:
        ood_accepted += 1
        ood_accepted_list.append(os.path.basename(p))

false_accept_rate = (ood_accepted / ood_total * 100) if ood_total>0 else None

# Save best chosen thresholds
with open('debug_outputs/selected_thresholds.txt','w',encoding='utf-8') as f:
    f.write(str(best))

# Save test & ood summary
with open('debug_outputs/grid_search_result_summary.txt','w',encoding='utf-8') as f:
    f.write('Best thresholds: '+str(best) + '\n')
    f.write('Test size: '+str(len(test_paths))+'\n')
    f.write('Test accepted: '+str(accepted_count)+'\n')
    f.write('Test rejected: '+str(rejected_count)+'\n')
    f.write('Material acc on accepted: '+str(material_acc_on_accepted)+'\n')
    f.write('Goddess acc on accepted: '+str(goddess_acc_on_accepted)+'\n')
    f.write('OOD total: '+str(ood_total)+'\n')
    f.write('OOD accepted: '+str(ood_accepted)+'\n')
    f.write('False accept rate (%): '+str(false_accept_rate)+'\n')
    f.write('OOD accepted list: '+str(ood_accepted_list)+'\n')

print('\nGRID SEARCH COMPLETE. Summary:')
print('Best thresholds:',best)
print('Test accepted:',accepted_count,'rejected:',rejected_count)
print('Material acc on accepted:',material_acc_on_accepted)
print('Goddess acc on accepted:',goddess_acc_on_accepted)
print('OOD total:',ood_total,'OOD accepted:',ood_accepted,'false_accept_rate(%):',false_accept_rate)
print('Detailed CSV: debug_outputs/threshold_grid_search.csv')
print('Summary: debug_outputs/grid_search_result_summary.txt')
