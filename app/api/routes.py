from __future__ import annotations

import logging
from time import perf_counter

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.schemas import AnalyzeResponse, HealthResponse, PredictResponse, ReportRequest, ReportResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import InvalidImageError, ModelUnavailableError
from app.services.analysis_service import build_analysis, parse_history
from app.services.image_io import load_heatmap_array, load_image_array, load_mask_array
from app.services.model_registry import ModelRegistry
from app.services.report_service import ReportService

LOGGER = logging.getLogger(__name__)
router = APIRouter()
_registry: ModelRegistry | None = None
_report_service: ReportService | None = None


def configure_services(settings: Settings) -> None:
    global _registry, _report_service
    _registry = ModelRegistry(settings)
    _report_service = ReportService(settings)


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        configure_services(get_settings())
    return _registry


def get_report_service() -> ReportService:
    global _report_service
    if _report_service is None:
        configure_services(get_settings())
    return _report_service


@router.get('/health', response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings), registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    return HealthResponse(
        status='ok',
        app_name=settings.app_name,
        version=settings.app_version,
        classifiers_loaded=registry.classifiers_loaded,
        segmentation_available=registry.segmentation_available,
    )


@router.post('/predict', response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    registry: ModelRegistry = Depends(get_registry),
) -> PredictResponse:
    started = perf_counter()
    try:
        image_np = load_image_array(await image.read())
        analysis = build_analysis(image_np, image.filename or 'upload', registry)
        if analysis['prediction'] is None:
            raise HTTPException(status_code=503, detail='Classifier models are unavailable for prediction.')
        LOGGER.info('predict completed filename=%s duration=%.4fs', image.filename, perf_counter() - started)
        return PredictResponse(filename=analysis['filename'], quality=analysis['quality'], prediction=analysis['prediction'])
    except InvalidImageError as exc:
        LOGGER.warning('predict invalid image filename=%s', image.filename)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/analyze', response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    mask: UploadFile | None = File(default=None),
    heatmap: UploadFile | None = File(default=None),
    history: str | None = Form(default=None),
    registry: ModelRegistry = Depends(get_registry),
) -> AnalyzeResponse:
    started = perf_counter()
    try:
        image_np = load_image_array(await image.read())
        mask_np = load_mask_array(await mask.read()) if mask is not None else None
        heatmap_np = load_heatmap_array(await heatmap.read()) if heatmap is not None else None
        history_rows = parse_history(history)
        analysis = build_analysis(image_np, image.filename or 'upload', registry, mask_np=mask_np, heatmap_np=heatmap_np, history=history_rows)
        LOGGER.info('analyze completed filename=%s duration=%.4fs warnings=%d', image.filename, perf_counter() - started, len(analysis['warnings']))
        return AnalyzeResponse(**analysis)
    except InvalidImageError as exc:
        LOGGER.warning('analyze invalid payload filename=%s', image.filename)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/report', response_model=ReportResponse)
def report(
    request: ReportRequest,
    report_service: ReportService = Depends(get_report_service),
) -> ReportResponse:
    try:
        report_text = report_service.generate(request.analysis, patient_name=request.patient_name)
        return ReportResponse(report=report_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
