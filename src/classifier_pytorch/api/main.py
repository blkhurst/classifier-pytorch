"""Run locally with: uvicorn classifier_pytorch.api.main:app --app-dir src --port 8000"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Path

from classifier_pytorch.mnist import load_sample
from classifier_pytorch.training import TrainingConfig
from .images import prepare_drawing
from .schemas import DrawingRequest, TrainRequest
from .session import TrainingSession


def create_app(session: TrainingSession | None = None) -> FastAPI:
    # Own One Session For The Lifetime Of The Server
    @asynccontextmanager
    async def lifespan(app):
        app.state.session = session if session is not None else TrainingSession()
        try:
            yield
        finally:
            app.state.session.close()

    app = FastAPI(title="classifier-pytorch", lifespan=lifespan)

    @app.get("/api/status")
    def status():
        return app.state.session.status()

    @app.post("/api/train", status_code=202)
    def start_training(request: TrainRequest):
        try:
            return app.state.session.start(TrainingConfig(**request.model_dump()))
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/train/stop")
    def stop_training():
        return app.state.session.request_stop()

    @app.post("/api/predict")
    def predict_drawing(request: DrawingRequest):
        try:
            image = prepare_drawing(request.image)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return app.state.session.prediction(image)

    @app.get("/api/samples/{index}")
    def sample(index: int = Path(ge=0, lt=60_000)):
        try:
            image, label = load_sample(index, app.state.session.data_dir)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        return {**app.state.session.prediction(image), "label": label, "sample": index}

    return app


app = create_app()
