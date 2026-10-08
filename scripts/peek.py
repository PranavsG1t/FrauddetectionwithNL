import pandas as pd, cv2, numpy as np
m = pd.read_csv("outputs/document_dataset/manifest.csv")
sid = m[m.split == "test"].source_id.iloc[0]
rows = m[m.source_id == sid]
imgs = [cv2.resize(cv2.imread(p), (224, 224)) for p in rows.path]
big = [cv2.imread(p) for p in rows.path]
cv2.imwrite("outputs/peek_224.png", np.hstack(imgs))
cv2.imwrite("outputs/peek_full.png", np.hstack([cv2.resize(b, (600, 400)) for b in big]))
print(rows[["path", "label"]])