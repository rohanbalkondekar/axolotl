# Upload Checkpoint to HuggingFace
from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="./finetuned-model/merged",
    repo_id="rohanbalkondekar/MistralOrca-7B-BankingSupport",
    repo_type="model",
)