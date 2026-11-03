import os
import csv
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Settings (consistent with app.py)
BASE_PATH = "Goddess_Split"
VALIDATION_PATH = os.path.join(BASE_PATH, "validation")
TEST_PATH = os.path.join(BASE_PATH, "test")
NEW_TEST_IMAGES = "New_Test_Images"
MODEL_PATH = os.path.join(BASE_PATH, "goddess_material_resnet18_v3_best.pth")
IMAGE_SIZE = 224
BATCH_SIZE = 16
RECOGNITION_THRESHOLD = 0.60
CENTROID_THRESHOLD = 0.50

GODDESSES = [
    "Annapurna",
    "Durga",
    "Kali",
    "Lakshmi",
    "Lalitha",
    "Saraswati"
]

MATERIALS = [
    "Brass",
    "Bronze",
    "Copper",
    "Silver",
    "Stone",
    "Wood"
]

GODDESS_TO_INDEX = {name: i for i, name in enumerate(GODDESSES)}
MATERIAL_TO_INDEX = {name: i for i, name in enumerate(MATERIALS)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Dataset loader (validation/test have same layout as in repo)
class GoddessMaterialDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for current_root, folders, files in os.walk(root_dir):
            for file in files:
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                image_path = os.path.join(current_root, file)
                relative_path = os.path.relpath(image_path, root_dir)
                path_parts = relative_path.split(os.sep)
                goddess = None
                material = None
                for name in GODDESSES:
                    if name in path_parts:
                        goddess = name
                        break
                for name in MATERIALS:
                    if name in path_parts:
                        material = name
                        break
                if goddess is not None and material is not None:
                    goddess_label = GODDESS_TO_INDEX[goddess]
                    material_label = MATERIAL_TO_INDEX[material]
                    self.samples.append((image_path, goddess_label, material_label))
        print("Loaded", len(self.samples), "images from", root_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, goddess_label, material_label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, goddess_label, material_label, image_path


# Create model same as Streamlit
model = models.resnet18(weights=None)
number_of_features = model.fc.in_features
model.fc = nn.Identity()
model.goddess_classifier = nn.Linear(number_of_features, len(GODDESSES))
model.material_classifier = nn.Linear(number_of_features, len(MATERIALS))
model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded from", MODEL_PATH)


def extract_feature(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        features = model(image_tensor)
        return features.cpu().numpy().reshape(-1)


def predict(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        features = model(image_tensor)
        goddess_out = model.goddess_classifier(features)
        material_out = model.material_classifier(features)
        goddess_prob = torch.softmax(goddess_out, dim=1).cpu().numpy().reshape(-1)
        material_prob = torch.softmax(material_out, dim=1).cpu().numpy().reshape(-1)
        goddess_idx = int(np.argmax(goddess_prob))
        material_idx = int(np.argmax(material_prob))
        goddess_conf = float(np.max(goddess_prob))
        material_conf = float(np.max(material_prob))
        feat = features.cpu().numpy().reshape(-1)
        return goddess_idx, goddess_conf, material_idx, material_conf, feat


# Build validation features and centroids per material
val_dataset = GoddessMaterialDataset(VALIDATION_PATH, transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

material_features = {i: [] for i in range(len(MATERIALS))}

print("Computing validation features for centroids...")
for images, goddess_labels, material_labels, paths in val_loader:
    for i in range(images.size(0)):
        img = images[i]
        mlabel = int(material_labels[i])
        feat = extract_feature(img)
        material_features[mlabel].append(feat)

material_centroids = {}
for m, feats in material_features.items():
    if len(feats) == 0:
        material_centroids[m] = np.zeros(number_of_features)
    else:
        material_centroids[m] = np.mean(np.stack(feats, axis=0), axis=0)

# Normalize centroids for cosine similarity
centroid_matrix = np.stack([material_centroids[i] for i in range(len(MATERIALS))], axis=0)
centroid_norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
centroid_matrix_normed = centroid_matrix / np.clip(centroid_norms, 1e-8, None)

# Helper to compute OOD score: 1 - max cosine similarity to any material centroid
def ood_score_from_feature(feature_vec):
    f = feature_vec.reshape(1, -1)
    fnorm = f / (np.linalg.norm(f) + 1e-8)
    sims = cosine_similarity(fnorm, centroid_matrix_normed)[0]
    max_sim = float(np.max(sims))
    return 1.0 - max_sim, max_sim


# Select several known valid test images (take all test images but cap to N for speed)
test_dataset = GoddessMaterialDataset(TEST_PATH, transform)
# Use all test images
valid_results = []
print("Running inference on test images (valid sculptures)...")
for idx in range(len(test_dataset)):
    img_tensor, true_g, true_m, path = test_dataset[idx]
    pg, cg, pm, cm, feat = predict(img_tensor)
    dist_to_pred = np.linalg.norm(feat - material_centroids[pm])
    dist_to_true = np.linalg.norm(feat - material_centroids[true_m])
    ood_score, max_sim = ood_score_from_feature(feat)
    accept = (cg >= RECOGNITION_THRESHOLD) and (cm >= RECOGNITION_THRESHOLD) and (max_sim >= CENTROID_THRESHOLD)
    valid_results.append({
        'path': path,
        'true_goddess': GODDESSES[true_g],
        'true_material': MATERIALS[true_m],
        'pred_goddess': GODDESSES[pg],
        'pred_material': MATERIALS[pm],
        'goddess_conf': cg,
        'material_conf': cm,
        'dist_to_pred_material': float(dist_to_pred),
        'dist_to_true_material': float(dist_to_true),
        'max_centroid_similarity': float(max_sim),
        'ood_score': float(ood_score),
        'accepted': bool(accept)
    })

# Now run OOD images (use New_Test_Images directory)
ood_images = []
if os.path.exists(NEW_TEST_IMAGES):
    for f in sorted(os.listdir(NEW_TEST_IMAGES)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            ood_images.append(os.path.join(NEW_TEST_IMAGES, f))
print("Found OOD images:", len(ood_images))
ood_results = []
for path in ood_images:
    try:
        image = Image.open(path).convert('RGB')
        img_tensor = transform(image)
        pg, cg, pm, cm, feat = predict(img_tensor)
        dist_to_pred = np.linalg.norm(feat - material_centroids[pm])
        ood_score, max_sim = ood_score_from_feature(feat)
        accept = (cg >= RECOGNITION_THRESHOLD) and (cm >= RECOGNITION_THRESHOLD) and (max_sim >= CENTROID_THRESHOLD)
        ood_results.append({
            'path': path,
            'pred_goddess': GODDESSES[pg],
            'pred_material': MATERIALS[pm],
            'goddess_conf': cg,
            'material_conf': cm,
            'dist_to_pred_material': float(dist_to_pred),
            'max_centroid_similarity': float(max_sim),
            'ood_score': float(ood_score),
            'accepted': bool(accept)
        })
    except Exception as e:
        print('Error processing', path, e)

# Save detailed CSVs
os.makedirs('debug_outputs', exist_ok=True)
with open('debug_outputs/valid_test_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = list(valid_results[0].keys()) if valid_results else []
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in valid_results:
        writer.writerow(row)

with open('debug_outputs/ood_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = list(ood_results[0].keys()) if ood_results else []
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in ood_results:
        writer.writerow(row)

# Summarize valid images acceptance vs rejection
valid_total = len(valid_results)
valid_accepted = sum(1 for r in valid_results if r['accepted'])
valid_rejected = valid_total - valid_accepted
false_rejection_rate = (valid_rejected / valid_total * 100) if valid_total>0 else 0.0

# Summarize OOD acceptance vs rejection
ood_total = len(ood_results)
ood_accepted = sum(1 for r in ood_results if r['accepted'])
ood_rejected = ood_total - ood_accepted
false_acceptance_rate = (ood_accepted / ood_total * 100) if ood_total>0 else 0.0

print('\n--- VALID TEST IMAGES SUMMARY ---')
print('Valid images tested:', valid_total)
print('Valid images accepted:', valid_accepted)
print('Valid images rejected:', valid_rejected)
print('False rejection rate: {:.2f}%'.format(false_rejection_rate))

print('\n--- OOD IMAGES SUMMARY ---')
print('OOD images tested:', ood_total)
print('OOD images rejected:', ood_rejected)
print('OOD images accepted:', ood_accepted)
print('False acceptance rate: {:.2f}%'.format(false_acceptance_rate))

# Now run a threshold sweep on VALIDATION set to find tradeoffs
print('\nRunning threshold sweep on VALIDATION set...')
thresholds = np.arange(0.30, 0.96, 0.05)
sweep = []

# Precompute validation predictions
val_preds = []
for idx in range(len(val_dataset)):
    img, true_g, true_m, path = val_dataset[idx]
    pg, cg, pm, cm, feat = predict(img)
    val_preds.append({'true_g': true_g, 'true_m': true_m, 'pg': pg, 'cg': cg, 'pm': pm, 'cm': cm})

for thr in thresholds:
    total = len(val_preds)
    accepted = [p for p in val_preds if (p['cg'] >= thr and p['cm'] >= thr)]
    rejected = total - len(accepted)
    # compute accuracy on accepted
    if len(accepted) > 0:
        g_acc = sum(1 for p in accepted if p['true_g'] == p['pg']) / len(accepted)
        m_acc = sum(1 for p in accepted if p['true_m'] == p['pm']) / len(accepted)
    else:
        g_acc = None
        m_acc = None
    sweep.append({'threshold': float(thr), 'accepted_count': len(accepted), 'rejected_count': rejected, 'goddess_acc_on_accepted': g_acc, 'material_acc_on_accepted': m_acc})

# Save sweep results
with open('debug_outputs/threshold_sweep.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = list(sweep[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in sweep:
        writer.writerow(row)

print('\nThreshold sweep saved to debug_outputs/threshold_sweep.csv')
print('Top few sweep results:')
for row in sweep[:5]:
    print(row)

print('\nDiagnostics complete. Detailed CSVs in debug_outputs/.')
