import joblib
import cloudpickle
import os

def save_pipeline_summary(
    final_result,
    model,
    best_algo_clean_data,
    best_algo_tame_outlier,
    best_algo_normalization,
    best_algo_feature_selection,
    best_algo_model_training,
    best_accuracy_model_training,
    file_path
):
    with open(file_path, "w") as f:
        f.write(f"{final_result}\n\n")
        f.write(f"Model Config:\n{model}\n\n")
        f.write(f"Final_Acc: {best_accuracy_model_training}\n\n")
        f.write(f"clean: {best_algo_clean_data}\n")
        f.write(f"outlier: {best_algo_tame_outlier}\n")
        f.write(f"norm: {best_algo_normalization}\n")
        f.write(f"feature: {best_algo_feature_selection}\n")
        f.write(f"model: {best_algo_model_training}\n")

    print(f"Summary saved to {file_path}")

path_to_dir = path.split("/")
path_to_dir.pop()
new_path = "/".join(path_to_dir)

joblib.dump(model, os.path.join(new_path, "model.pkl"))

imputer = best_algo_class_clean_data

with open(os.path.join(new_path, "clean_data.pkl"), "wb") as f:
    cloudpickle.dump(imputer, f)

joblib.dump(all_mappings, os.path.join(new_path, "all_mapping.pkl"))

joblib.dump(x_copy_feature_selection.columns.tolist(), os.path.join(new_path, "features.pkl"))