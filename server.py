import io
import json
import os
import uuid
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import soundfile as sf
import torch
import whisper
import boto3
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from db import SessionLocal, Base, engine
from models import User, Segment as SegmentModel, Upload
from auth import hash_password, verify_password, create_access_token, require_auth

SAMPLE_RATE = 16000

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_PUBLIC_URL_PREFIX = os.getenv("S3_PUBLIC_URL_PREFIX", "")

@dataclass
class WordInfo:
    word: str
    start: float
    end: float


@dataclass
class Segment:
    source: str
    start_sec: float
    end_sec: float
    text: str
    confidence: float
    translated: Optional[str] = None
    words: List[WordInfo] = field(default_factory=list)


class ModelPool:
    def __init__(self):
        self.cache: Dict[Tuple[str, str], List[Tuple[str, whisper.Whisper]]] = {}

    def get_pool(self, model_size: str, devices: List[str]):
        key = (model_size, ",".join(devices))
        if key not in self.cache:
            models = []
            for d in devices:
                device = torch.device(d)
                models.append((d, whisper.load_model(model_size, device=device)))
            self.cache[key] = models
        return self.cache[key]

    def get(self, model_size: str, devices: List[str], key: str):
        pool = self.get_pool(model_size, devices)
        idx = abs(hash(key)) % len(pool)
        return pool[idx]


POOL = ModelPool()


class Translator:
    def __init__(self, target_lang: str):
        self.target_lang = target_lang
        self.argos_available = False
        if target_lang and target_lang != "en":
            try:
                from argostranslate import translate as argos_translate
                self.argos_translate = argos_translate
                self.argos_available = True
            except Exception:
                self.argos_available = False

    def translate_text(self, text: str) -> Optional[str]:
        if not self.target_lang:
            return None
        if self.target_lang == "en":
            return None
        if not self.argos_available:
            return None
        try:
            return self.argos_translate.translate(text, "auto", self.target_lang)
        except Exception:
            return None


def word_align_whisperx(audio_np: np.ndarray, text: str, language: Optional[str], device: str):
    try:
        import whisperx
    except Exception:
        return []

    if not text:
        return []

    try:
        align_model, metadata = whisperx.load_align_model(language_code=language or "en", device=device)
        result = {"segments": [{"text": text, "start": 0.0, "end": len(audio_np) / SAMPLE_RATE}]}
        aligned = whisperx.align(result, align_model, metadata, audio_np, SAMPLE_RATE, device=device)
        words = []
        for seg in aligned["segments"]:
            for w in seg.get("words", []):
                if w.get("word"):
                    words.append(WordInfo(w["word"], w.get("start", 0.0), w.get("end", 0.0)))
        return words
    except Exception:
        return []


def decode_with_confidence(model, audio_np: np.ndarray, language: Optional[str], task: str = "transcribe"):
    audio_np = whisper.pad_or_trim(audio_np)
    mel = whisper.log_mel_spectrogram(audio_np).to(model.device)

    if language == "auto":
        options = whisper.DecodingOptions(fp16=(model.device.type == "cuda"), task=task)
    else:
        options = whisper.DecodingOptions(language=language, fp16=(model.device.type == "cuda"), task=task)

    result = whisper.decode(model, mel, options)
    text = result.text.strip()
    confidence = float(np.exp(result.avg_logprob)) if result.avg_logprob is not None else 0.0
    return text, confidence


def to_srt_time(seconds: float) -> str:
    ms = int(seconds * 1000)
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms = ms % 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def build_srt_text(segments: List[Segment]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{to_srt_time(seg.start_sec)} --> {to_srt_time(seg.end_sec)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def s3_client():
    if S3_ENDPOINT:
        return boto3.client("s3", region_name=AWS_REGION, endpoint_url=S3_ENDPOINT)
    return boto3.client("s3", region_name=AWS_REGION)


def upload_to_s3(file_bytes: bytes, filename: str) -> str:
    if not S3_BUCKET:
        raise HTTPException(status_code=500, detail="S3_BUCKET not configured")

    key = f"uploads/{uuid.uuid4()}_{filename}"
    client = s3_client()
    client.put_object(Bucket=S3_BUCKET, Key=key, Body=file_bytes)

    if S3_PUBLIC_URL_PREFIX:
        return f"{S3_PUBLIC_URL_PREFIX}/{key}"
    return f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"


def create_presigned_upload_url(filename: str, content_type: str) -> Dict[str, Any]:
    if not S3_BUCKET:
        raise HTTPException(status_code=500, detail="S3_BUCKET not configured")

    key = f"uploads/{uuid.uuid4()}_{filename}"
    client = s3_client()
    url = client.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": S3_BUCKET, "Key": key, "ContentType": content_type},
        ExpiresIn=3600,
    )

    if S3_PUBLIC_URL_PREFIX:
        public_url = f"{S3_PUBLIC_URL_PREFIX}/{key}"
    else:
        public_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{key}"

    return {"upload_url": url, "key": key, "public_url": public_url}


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

SEGMENTS: Dict[str, List[Segment]] = {"MIC": [], "FILE": []}
WS_CLIENTS: set = set()


@app.post("/auth/register")
def register(username: str = Form(...), password: str = Form(...), email: str = Form(""), db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed = hash_password(password)
    user = User(username=username, email=email or None, hashed_password=hashed)
    db.add(user)
    db.commit()
    return {"status": "ok"}


@app.post("/auth/token")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.username)
    return {"access_token": token, "token_type": "bearer"}


def get_current_user(db: Session, username: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@app.post("/s3/presign")
def presign_upload(
    filename: str = Form(...),
    content_type: str = Form("audio/wav"),
    user: str = Depends(require_auth),
):
    return create_presigned_upload_url(filename, content_type)


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    source: str = Form("FILE"),
    model_size: str = Form("small"),
    language: str = Form("auto"),
    translate_target: str = Form(""),
    align: bool = Form(False),
    devices: str = Form("cuda:0"),
    save_to_s3: bool = Form(False),
    user: str = Depends(require_auth),
    db: Session = Depends(get_db),
):
    devices_list = [d.strip() for d in devices.split(",") if d.strip()]
    device_str, model = POOL.get(model_size, devices_list, source)
    translator = Translator(translate_target)

    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if sr != SAMPLE_RATE:
        raise HTTPException(status_code=400, detail="Audio must be 16kHz")
    if audio.ndim > 1:
        audio = audio[:, 0]

    if save_to_s3:
        s3_url = upload_to_s3(audio_bytes, file.filename)
        current = get_current_user(db, user)
        db.add(Upload(user_id=current.id, filename=file.filename, s3_key=s3_url, s3_url=s3_url))
        db.commit()

    text, conf = decode_with_confidence(model, audio, language, task="transcribe")
    translated = None
    if translate_target:
        if translate_target == "en":
            translated, _ = decode_with_confidence(model, audio, language, task="translate")
        else:
            translated = translator.translate_text(text)

    words = word_align_whisperx(audio, text, None if language == "auto" else language, device_str) if align else []
    seg = Segment(source, 0.0, len(audio) / SAMPLE_RATE, text, conf, translated, words)
    SEGMENTS[source].append(seg)

    current = get_current_user(db, user)
    db_seg = SegmentModel(
        user_id=current.id,
        source=source,
        start_sec=seg.start_sec,
        end_sec=seg.end_sec,
        text=seg.text,
        confidence=seg.confidence,
        translated=seg.translated,
        words_json=json.dumps([asdict(w) for w in seg.words], ensure_ascii=False) if seg.words else None,
    )
    db.add(db_seg)
    db.commit()

    payload = asdict(seg)
    for ws in list(WS_CLIENTS):
        try:
            await ws.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

    return JSONResponse(payload)


STREAMS: Dict[str, Dict] = {}


@app.post("/stream/start")
def stream_start(
    source: str = Form("MIC"),
    model_size: str = Form("small"),
    language: str = Form("auto"),
    translate_target: str = Form(""),
    align: bool = Form(False),
    devices: str = Form("cuda:0"),
    chunk_sec: float = Form(3.0),
    user: str = Depends(require_auth),
):
    session_id = str(uuid.uuid4())
    STREAMS[session_id] = {
        "source": source,
        "model_size": model_size,
        "language": language,
        "translate_target": translate_target,
        "align": align,
        "devices": devices,
        "chunk_sec": float(chunk_sec),
        "buffer": np.zeros((0,), dtype=np.float32),
        "time_cursor": 0.0,
        "user": user,
    }
    return {"session_id": session_id}


@app.post("/stream/chunk")
async def stream_chunk(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    user: str = Depends(require_auth),
    db: Session = Depends(get_db),
):
    if session_id not in STREAMS:
        raise HTTPException(status_code=404, detail="Invalid session")

    state = STREAMS[session_id]
    devices_list = [d.strip() for d in state["devices"].split(",") if d.strip()]
    device_str, model = POOL.get(state["model_size"], devices_list, session_id)
    translator = Translator(state["translate_target"])

    audio_bytes = await file.read()
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if sr != SAMPLE_RATE:
        raise HTTPException(status_code=400, detail="Audio must be 16kHz")
    if audio.ndim > 1:
        audio = audio[:, 0]

    state["buffer"] = np.concatenate([state["buffer"], audio], axis=0)
    chunk_samples = int(SAMPLE_RATE * state["chunk_sec"])

    produced = []
    while len(state["buffer"]) >= chunk_samples:
        chunk = state["buffer"][:chunk_samples]
        state["buffer"] = state["buffer"][chunk_samples:]

        start = state["time_cursor"]
        end = start + state["chunk_sec"]
        state["time_cursor"] += state["chunk_sec"]

        text, conf = decode_with_confidence(model, chunk, state["language"], task="transcribe")
        if not text:
            continue

        translated = None
        if state["translate_target"]:
            if state["translate_target"] == "en":
                translated, _ = decode_with_confidence(model, chunk, state["language"], task="translate")
            else:
                translated = translator.translate_text(text)

        words = word_align_whisperx(chunk, text, None if state["language"] == "auto" else state["language"], device_str) if state["align"] else []
        seg = Segment(state["source"], start, end, text, conf, translated, words)
        SEGMENTS[state["source"]].append(seg)
        produced.append(asdict(seg))

        current = get_current_user(db, state["user"])
        db_seg = SegmentModel(
            user_id=current.id,
            source=state["source"],
            start_sec=seg.start_sec,
            end_sec=seg.end_sec,
            text=seg.text,
            confidence=seg.confidence,
            translated=seg.translated,
            words_json=json.dumps([asdict(w) for w in seg.words], ensure_ascii=False) if seg.words else None,
        )
        db.add(db_seg)
        db.commit()

        payload = json.dumps(asdict(seg), ensure_ascii=False)
        for ws in list(WS_CLIENTS):
            try:
                await ws.send_text(payload)
            except Exception:
                pass

    return {"segments": produced}


@app.post("/stream/finish")
def stream_finish(session_id: str = Form(...), user: str = Depends(require_auth)):
    STREAMS.pop(session_id, None)
    return {"status": "ok"}


@app.get("/segments")
def list_segments(source: str = "", q: str = "", user: str = Depends(require_auth), db: Session = Depends(get_db)):
    current = get_current_user(db, user)
    query = db.query(SegmentModel).filter(SegmentModel.user_id == current.id)
    if source:
        query = query.filter(SegmentModel.source == source)
    if q:
        query = query.filter(SegmentModel.text.ilike(f"%{q}%"))
    rows = query.order_by(SegmentModel.created_at.desc()).all()

    data = []
    for r in rows:
        words = json.loads(r.words_json) if r.words_json else []
        data.append({
            "source": r.source,
            "start_sec": r.start_sec,
            "end_sec": r.end_sec,
            "text": r.text,
            "confidence": r.confidence,
            "translated": r.translated,
            "words": words,
        })
    return JSONResponse(data)


@app.get("/download/srt")
def download_srt(source: str = "", user: str = Depends(require_auth), db: Session = Depends(get_db)):
    current = get_current_user(db, user)
    query = db.query(SegmentModel).filter(SegmentModel.user_id == current.id)
    if source:
        query = query.filter(SegmentModel.source == source)
    rows = query.order_by(SegmentModel.created_at.asc()).all()

    segs = [Segment(r.source, r.start_sec, r.end_sec, r.text, r.confidence, r.translated, []) for r in rows]
    return PlainTextResponse(build_srt_text(segs), media_type="text/plain")


@app.get("/download/json")
def download_json(source: str = "", user: str = Depends(require_auth), db: Session = Depends(get_db)):
    current = get_current_user(db, user)
    query = db.query(SegmentModel).filter(SegmentModel.user_id == current.id)
    if source:
        query = query.filter(SegmentModel.source == source)
    rows = query.order_by(SegmentModel.created_at.asc()).all()

    data = []
    for r in rows:
        words = json.loads(r.words_json) if r.words_json else []
        data.append({
            "source": r.source,
            "start_sec": r.start_sec,
            "end_sec": r.end_sec,
            "text": r.text,
            "confidence": r.confidence,
            "translated": r.translated,
            "words": words,
        })
    return JSONResponse(data)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    token = ws.query_params.get("token", "")
    if not token:
        await ws.close(code=4401)
        return

    await ws.accept()
    WS_CLIENTS.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        WS_CLIENTS.remove(ws)