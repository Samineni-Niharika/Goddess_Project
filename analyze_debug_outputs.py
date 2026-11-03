import csv
import numpy as np

path = 'debug_outputs/valid_test_results.csv'
rows = []
with open(path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

accepted = [float(r['max_centroid_similarity']) for r in rows if r['accepted']=='True']
rejected = [float(r['max_centroid_similarity']) for r in rows if r['accepted']=='False']

print('Accepted count:', len(accepted))
print('Rejected count:', len(rejected))
if len(accepted)>0:
    print('Accepted max_sim: mean {:.4f}, min {:.4f}, 25% {:.4f}, 50% {:.4f}, 75% {:.4f}'.format(np.mean(accepted), np.min(accepted), np.percentile(accepted,25), np.median(accepted), np.percentile(accepted,75)))
if len(rejected)>0:
    print('Rejected max_sim: mean {:.4f}, min {:.4f}, 25% {:.4f}, 50% {:.4f}, 75% {:.4f}'.format(np.mean(rejected), np.min(rejected), np.percentile(rejected,25), np.median(rejected), np.percentile(rejected,75)))

# Also print overlap count: how many rejected have max_sim >= median of accepted
if len(accepted)>0 and len(rejected)>0:
    med_accepted = np.median(accepted)
    overlap = sum(1 for r in rejected if r>=med_accepted)
    print('Rejected images with max_sim >= median(accepted):', overlap)

# Print a few lowest accepted and highest rejected to inspect
accepted_sorted = sorted(accepted)
rejected_sorted = sorted(rejected, reverse=True)
print('\nLowest 5 accepted max_sim:', accepted_sorted[:5])
print('Highest 5 rejected max_sim:', rejected_sorted[:5])
