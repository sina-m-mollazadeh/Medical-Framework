import joblib
import cloudpickle
import os

def save_pipeline_summary(
    final_result,
    model,
    best_algo_clean,
    best_algo_tame,
    best_algo_normalization,
    best_algo_feature,
    best_algo_model,
    best_accuracy_model,
    file_path
):
    with open(file_path, "w") as f:
        f.write(f"{final_result}\n\n")
        f.write(f"Model Config:\n{model}\n\n")
        f.write(f"Final_Acc: {best_accuracy_model}\n\n")
        f.write(f"clean: {best_algo_clean}\n")
        f.write(f"outlier: {best_algo_tame}\n")
        f.write(f"norm: {best_algo_normalization}\n")
        f.write(f"feature: {best_algo_feature}\n")
        f.write(f"model: {best_algo_model}\n")

    print(f"Summary saved to {file_path}")

