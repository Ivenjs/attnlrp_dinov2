import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path_new = "/workspaces/vast-gorilla/gorillawatch/data2/stratified_open_split_NEW/spac23+24-body_face-squared-deduplicated"
data_path_old = "/workspaces/vast-gorilla/gorillawatch/data/eval_body_squared_cleaned_open_2024_bigval"
relevance_path = "/workspaces/vast-gorilla/gorillawatch/iven_thesis/relevance_db/ViTG-body_face-spac23+24_spac23+24-body_face-squared-deduplicated_test_soft_knn_margin_db.pt"
knn_path_test = "/workspaces/vast-gorilla/gorillawatch/iven_thesis/knn_db/ViTG-body_face-spac23+24_spac23+24-body_face-squared-deduplicated_test_db.pt"
knn_path_all = "/workspaces/vast-gorilla/gorillawatch/iven_thesis/knn_db/ViTG-body_face-spac23+24_all_dataset_train+test_db.pt"

db_data_test = torch.load(knn_path_test, map_location=device, weights_only=False)
db_data_all = torch.load(knn_path_all, map_location=device, weights_only=False)
#db_data["embeddings"], db_data["labels"], db_data["filenames"], db_data["videos"]



db_relevance = torch.load(relevance_path, map_location=device, weights_only=False)
"""
result_item = {
                        "filename": filename,
                        "params": {
                            "conv_gamma": conv_gamma,
                            "lin_gamma": lin_gamma,
                            "distance_metric": distance_metric,
                            "proxy_temp": proxy_temp,
                            "topk": topk,
                        },
                        "mode": mode,
                        "relevance": relevance_single.detach().cpu(),
                        "mask": mask_single,
                        **extra_info
                    }
"""

#in the datapaths look resursively for files that containe "R066_20221118_155" in their name and count occurences
print("Looking for files containing 'R066_20221118_155' in their name...")
print("------------------ new data path ------------------")
target_string = "R066_20221118_155"
for dirpath, dirnames, filenames in os.walk(data_path_new):
    for filename in filenames:
        if target_string in filename:
            print(os.path.join(dirpath, filename))
print("-------------------- old data path ------------------")
for dirpath, dirnames, filenames in os.walk(data_path_old):
    for filename in filenames:
        if target_string in filename:
            print(os.path.join(dirpath, filename))

print("Looking in knn db...")
print("Test db:")
test_filenames = db_data_test["filenames"]
for filename in test_filenames:
    if target_string in filename:
        print(os.path.join(knn_path_test, filename))
print("All db:")
all_filenames = db_data_all["filenames"]
for filename in all_filenames:
    if target_string in filename:
        print(os.path.join(knn_path_all, filename))

print("Looking in relevance db...")
relevance_filenames = [item["filename"] for item in db_relevance]
for filename in relevance_filenames:
    if target_string in filename:
        print(os.path.join(relevance_path, filename))