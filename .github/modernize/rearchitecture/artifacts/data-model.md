# Data Model

## Entities

- Image sample: `(image_path, goddess_label, material_label)` in `GoddessDataset`.
- Goddess label: one of Annapurna, Durga, Kali, Lakshmi, Lalitha, Saraswati; integer range 0–5.
- Material label: one of Brass, Bronze, Copper, Silver, Stone, Wood; integer range 0–5.
- Feature vector: ResNet18 penultimate representation used by both classifier heads and centroid checks.
- Goddess logits/probabilities: six-dimensional output and calibrated softmax distribution.
- Material logits/probabilities: six-dimensional output and calibrated softmax distribution.
- Material centroid: mean validation feature vector for each material, row-normalized for cosine comparison.

## Relationships

- An image sample has exactly one goddess label and one material label.
- One shared feature vector feeds both the goddess and material classifiers.
- Each material centroid aggregates validation feature vectors belonging to that material.
- A prediction is accepted only when both calibrated classifier confidences and feature similarity pass thresholds.

## Dataset topology

`Goddess_Resized/<goddess>/<Metals>/<material>/image.jpg` is used for metal classes; Stone and Wood omit the `Metals` directory. `Goddess_Split/<split>/...` mirrors this layout for train, validation, and test.

Recorded project result script: 798 images, split as 558 train, 119 validation, and 121 test.
