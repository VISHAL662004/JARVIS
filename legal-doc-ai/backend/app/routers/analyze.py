from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.config import settings
from app.schemas.analysis import AnalyzeResponse, ChatAnswer, ChatQuestion, JobStatus
from app.services.analysis_pipeline import analysis_pipeline
from app.services.document_chat import document_chat_service, document_chat_store
from app.utils.job_store import job_store
from app.utils.validation import validate_pdf_upload

router = APIRouter(prefix="/analyze", tags=["analysis"])


async def _process_job(job_id: str, payload: bytes) -> None:
    await job_store.mark_running(job_id)
    try:
        loop = asyncio.get_running_loop()
        result, context = await loop.run_in_executor(None, analysis_pipeline.run_with_context, payload)
        await document_chat_store.register(job_id, context)
        await job_store.mark_completed(job_id, result)
    except Exception as exc:
        await job_store.mark_failed(job_id, str(exc))


@router.post("/upload", response_model=AnalyzeResponse)
async def analyze_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> AnalyzeResponse:
    await validate_pdf_upload(file)
    payload = await file.read()
    job = await job_store.create()
    background_tasks.add_task(_process_job, job.job_id, payload)
    return AnalyzeResponse(job_id=job.job_id, status_url=f"{settings.api_prefix}/analyze/jobs/{job.job_id}")


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/stream-summary")
async def stream_summary(file: UploadFile = File(...)) -> StreamingResponse:
    await validate_pdf_upload(file)
    payload = await file.read()

    async def event_stream():
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, analysis_pipeline.run, payload)
        for chunk in [result.summary_extractive, result.summary_abstractive]:
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/jobs/{job_id}/chat", response_model=ChatAnswer)
async def chat_with_document(job_id: str, payload: ChatQuestion) -> ChatAnswer:
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed" or not job.result:
        raise HTTPException(status_code=409, detail="Analysis is not complete yet")

    context = await document_chat_store.get(job_id)
    if not context:
        raise HTTPException(status_code=404, detail="Document context not available")

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: document_chat_service.answer(
            context=context,
            question=payload.question,
            analysis=job.result,
            history=[turn.model_dump() for turn in payload.history],
        ),
    )
