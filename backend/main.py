from fastapi import FastAPI

from backend.api.pose_gaze_routes import router as pose_gaze_router


app = FastAPI(
    title="Exam Proctoring System",
    version="0.1.0",
    description="Backend APIs for realtime exam supervision.",
)
app.include_router(pose_gaze_router)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}
