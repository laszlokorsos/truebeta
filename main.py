import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from web.router import router

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="TrueBeta", description="Dynamic stock beta estimation using Kalman filtering")

static_dir = os.path.join(BASE_DIR, "web", "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
