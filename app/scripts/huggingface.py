from huggingface_hub import HfApi

api = HfApi()
model_repo = "Janani-Withana/sinhala-farming-qa-model"

# Upload updated model folder
api.upload_folder(
    folder_path="models/fine_tuned_multilingual_model",
    repo_id=model_repo,
    commit_message="Updated fine-tuned model"
)

print(f"Model successfully updated at: https://huggingface.co/{model_repo}")

