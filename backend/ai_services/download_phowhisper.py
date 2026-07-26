from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="vinai/PhoWhisper-small",
    local_dir=r"D:\AI\Models\PhoWhisper-small",
    local_dir_use_symlinks=False,
)

print("Download completed!")