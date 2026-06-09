from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional, List

try:
    from modules.requests_tls import requests_session_with_os_trust
except Exception:
    import requests

    def requests_session_with_os_trust():
        return requests.Session()

from .asr_data import ASRDataSeg
from .asr_base import BaseASR


__version__ = "0.0.3"

API_BASE_URL = "https://member.bilibili.com/x/bcut/rubick-interface"

# 申请上传
API_REQ_UPLOAD = API_BASE_URL + "/resource/create"

# 提交上传
API_COMMIT_UPLOAD = API_BASE_URL + "/resource/create/complete"

# 创建任务
API_CREATE_TASK = API_BASE_URL + "/task"

# 查询结果
API_QUERY_RESULT = API_BASE_URL + "/task/result"
BCUT_MODEL_ID = "8"


class BcutASRError(RuntimeError):
    pass


def _compact_json(value: Any, limit: int = 1000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def _decode_response(resp: Any, action: str) -> dict[str, Any]:
    try:
        resp.raise_for_status()
    except Exception as exc:
        status = getattr(resp, "status_code", "unknown")
        body = str(getattr(resp, "text", "") or "")[:1000]
        raise BcutASRError(f"Bcut ASR {action} HTTP failed: status={status}; body={body}") from exc
    try:
        data = resp.json()
    except Exception as exc:
        status = getattr(resp, "status_code", "unknown")
        body = str(getattr(resp, "text", "") or "")[:1000]
        raise BcutASRError(f"Bcut ASR {action} returned non-JSON response: status={status}; body={body}") from exc
    if not isinstance(data, dict):
        raise BcutASRError(f"Bcut ASR {action} returned unexpected JSON: {_compact_json(data)}")
    return data


def _require_data(payload: dict[str, Any], action: str) -> Any:
    if "data" in payload and payload["data"] is not None:
        return payload["data"]
    code = payload.get("code")
    message = payload.get("message") or payload.get("msg")
    parts = ["missing data field"]
    if code is not None:
        parts.append(f"code={code}")
    if message:
        parts.append(f"message={message}")
    raise BcutASRError(f"Bcut ASR {action} failed: {', '.join(parts)}; response={_compact_json(payload)}")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _bcut_requests_session():
    session = requests_session_with_os_trust()
    # VPN clients often publish a system proxy that breaks Bcut's API and
    # presigned upload URLs. Keep this ASR provider direct by default.
    session.trust_env = _env_flag("BCUT_ASR_TRUST_ENV", default=False)
    return session


class BcutASR(BaseASR):
    """ 语音识别接口"""
    headers = {
        'User-Agent': 'Bilibili/1.0.0 (https://www.bilibili.com)',
        'Content-Type': 'application/json'
    }

    def __init__(self, audio_path: [str, bytes], use_cache: bool = False):
        super().__init__(audio_path, use_cache=use_cache)
        self.session = _bcut_requests_session()
        self.task_id: Optional[str] = None
        self.__etags: List[str] = []

        self.__in_boss_key: Optional[str] = None
        self.__resource_id: Optional[str] = None
        self.__upload_id: Optional[str] = None
        self.__upload_urls: List[str] = []
        self.__per_size: Optional[int] = None
        self.__clips: Optional[int] = None

        self.__download_url: Optional[str] = None

    @staticmethod
    def test_connection(timeout: int = 6) -> dict:
        try:
            session = _bcut_requests_session()
            resp = session.get(API_BASE_URL, timeout=timeout)
            ok = int(resp.status_code) < 500
            if ok:
                return {"success": True, "status_code": int(resp.status_code)}
            return {"success": False, "status_code": int(resp.status_code)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def upload(self) -> None:
        """申请上传"""
        if not self.file_binary:
            raise ValueError("none set data")
        payload = json.dumps({
            "type": 2,
            "name": "audio.mp3",
            "size": len(self.file_binary),
            "ResourceFileType": "mp3",
            "model_id": BCUT_MODEL_ID,
        })

        resp = self.session.post(
            API_REQ_UPLOAD,
            data=payload,
            headers=self.headers
        )
        resp = _decode_response(resp, "request upload")
        resp_data = _require_data(resp, "request upload")

        self.__in_boss_key = resp_data["in_boss_key"]
        self.__resource_id = resp_data["resource_id"]
        self.__upload_id = resp_data["upload_id"]
        self.__upload_urls = resp_data["upload_urls"]
        self.__per_size = resp_data["per_size"]
        self.__clips = len(resp_data["upload_urls"])

        logging.info(
            f"申请上传成功, 总计大小{resp_data['size'] // 1024}KB, {self.__clips}分片, 分片大小{resp_data['per_size'] // 1024}KB: {self.__in_boss_key}"
        )
        self.__upload_part()
        self.__commit_upload()

    def __upload_part(self) -> None:
        """上传音频数据"""
        for clip in range(self.__clips or 0):
            start_range = clip * (self.__per_size or 0)
            end_range = (clip + 1) * (self.__per_size or 0)
            logging.info(f"开始上传分片{clip}: {start_range}-{end_range}")
            resp = self.session.put(
                self.__upload_urls[clip],
                data=self.file_binary[start_range:end_range],
                headers=self.headers
            )
            try:
                resp.raise_for_status()
            except Exception as exc:
                status = getattr(resp, "status_code", "unknown")
                body = str(getattr(resp, "text", "") or "")[:1000]
                raise BcutASRError(f"Bcut ASR upload part failed: clip={clip}, status={status}; body={body}") from exc
            etag = resp.headers.get("Etag")
            if etag:
                self.__etags.append(etag)
            logging.info(f"分片{clip}上传成功: {etag}")

    def __commit_upload(self) -> None:
        """提交上传数据"""
        data = json.dumps({
            "InBossKey": self.__in_boss_key,
            "ResourceId": self.__resource_id,
            "Etags": ",".join(self.__etags),
            "UploadId": self.__upload_id,
            "model_id": BCUT_MODEL_ID,
        })
        resp = self.session.post(
            API_COMMIT_UPLOAD,
            data=data,
            headers=self.headers
        )
        resp = _decode_response(resp, "commit upload")
        resp_data = _require_data(resp, "commit upload")
        if not isinstance(resp_data, dict) or not resp_data.get("download_url"):
            raise BcutASRError(f"Bcut ASR commit upload failed: missing download_url; response={_compact_json(resp)}")
        self.__download_url = resp_data["download_url"]
        logging.info(f"提交成功")

    def create_task(self) -> str:
        """开始创建转换任务"""
        resp = self.session.post(
            API_CREATE_TASK, json={"resource": self.__download_url, "model_id": BCUT_MODEL_ID}, headers=self.headers
        )
        resp = _decode_response(resp, "create task")
        resp_data = _require_data(resp, "create task")
        if not isinstance(resp_data, dict) or not resp_data.get("task_id"):
            raise BcutASRError(f"Bcut ASR create task failed: missing task_id; response={_compact_json(resp)}")
        self.task_id = resp_data["task_id"]
        logging.info(f"任务已创建: {self.task_id}")
        return self.task_id

    def result(self, task_id: Optional[str] = None):
        """查询转换结果"""
        resp = self.session.get(API_QUERY_RESULT, params={"model_id": BCUT_MODEL_ID, "task_id": task_id or self.task_id}, headers=self.headers)
        resp = _decode_response(resp, "query result")
        return _require_data(resp, "query result")

    def _run(self, callback: Optional[callable] = None):
        if callback:
            try:
                callback(20, "正在上传音频到服务...")
            except Exception:
                pass
        self.upload()

        if callback:
            try:
                callback(40, "上传完成，创建识别任务...")
            except Exception:
                pass
        self.create_task()

        if callback:
            try:
                callback(55, "任务已创建，开始轮询结果...")
            except Exception:
                pass
        # 轮询检查任务状态
        task_resp = None
        for _ in range(500):
            task_resp = self.result()
            if not isinstance(task_resp, dict):
                raise BcutASRError(f"Bcut ASR query result returned invalid data: {_compact_json(task_resp)}")
            if task_resp.get("state") == 4:
                break
            if task_resp.get("state") in {-1, 5, 6}:
                raise BcutASRError(f"Bcut ASR task failed: {_compact_json(task_resp)}")
            time.sleep(1)
        else:
            raise BcutASRError(f"Bcut ASR task timed out after polling; task_id={self.task_id}")

        result_text = task_resp.get("result") if isinstance(task_resp, dict) else None
        if not result_text:
            raise BcutASRError(f"Bcut ASR task completed without result: {_compact_json(task_resp)}")

        if callback:
            try:
                callback(95, "转换完成，解析结果...")
            except Exception:
                pass
        logging.info(f"转换成功")
        try:
            return json.loads(result_text)
        except Exception as exc:
            raise BcutASRError(f"Bcut ASR result is not valid JSON: {_compact_json(result_text)}") from exc

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        return [ASRDataSeg(u.get('text') or u.get('transcript') or '', u['start_time'], u['end_time']) for u in resp_data['utterances']]
