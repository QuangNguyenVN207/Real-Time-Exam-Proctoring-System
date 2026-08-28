- Interpreter: `C:\Users\Wingery\cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
- Stack: Python, PyTorch, MediaPipe, OpenCV, XGBoost, Pandas, Parquet
- Code root: `backend/ai_services/pose_gaze`, Data root: `data/processed/`
- Always check imports, configs, artifacts before execution commands.
- Normalized frame coordinates [0.0, 1.0]. Avoid NaN/Infinity; use null/0.0 with masks for invalid values.

- backend/ai_services/pose_gaze và data/raw_video, data/processed/holistic_manifest, .github. Không đọc ngoài các folder 
- đọc one euro mapping.md, đọc NEW CHAT HANDOFF

- chạy video c3 c6 khi gọi $rows = Import-Csv data/processed/holistic_manifest.csv | Where-Object {
    $_.class_code -in @("c3", "c6") -and
    $_.media_readable -eq "True" -and
    $_.exclude_from_training -ne "1"
}

Push-Location backend/ai_services/pose_gaze
$env:PYTHONPATH = "../../../;."

foreach ($row in $rows) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($row.filename)
    $out = "pose_gaze/holistic/test_media/outputs/final_$stem.mp4"
    $json = "pose_gaze/holistic/test_media/outputs/final_$stem.json"

    Write-Host "START $($row.class_code) $($row.filename)"

    python -m pose_gaze.holistic.test_media $row.media_path `
        --model "../../../weights/yolov8n.pt" `
        --output $out `
        --landmarks-output $json `
        --no-display `
        --bbox-smoothing-alpha 0.5 `
        --crop-stabilization-alpha 0.8 `
        --face-hold-frames 3 `
        --face-fallback-model "../../../weights/mediapipe/face_landmarker.task"

    if ($LASTEXITCODE -eq 0) {
        Write-Host "DONE $($row.filename)"
    } else {
        Write-Host "FAIL $($row.filename) code=$LASTEXITCODE"
    }
}

Pop-Location