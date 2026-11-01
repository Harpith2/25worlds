import json
import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import service

import pytest
from fastapi.testclient import TestClient
from service import create_api, get_job_dir, MAX_BYTES


class FakeVolume:
    def __init__(self):
        self.commits=0
        self.reloads=0
        self.commit = Mock(side_effect=AssertionError("Blocking volume.commit called"))
        self.commit.aio = AsyncMock(side_effect=self._commit)
        self.reload = Mock(side_effect=AssertionError("Blocking volume.reload called"))
        self.reload.aio = AsyncMock(side_effect=self._reload)
    def _commit(self):
        self.commits+=1
    def _reload(self):
        self.reloads+=1


def create_sample_video(path: Path):
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-f','lavfi','-i','testsrc=size=320x240:rate=10:duration=8',
        '-c:v','mpeg4','-q:v','4',str(path)
    ], check=True, timeout=15)


def build_client(monkeypatch, tmp_path, spawn):
    monkeypatch.setenv('WORLD_ACCESS_CODE','long-test-code-2026')
    return TestClient(create_api(volume=FakeVolume(),enqueue=spawn,
                 root=tmp_path))


def test_access_required(monkeypatch, tmp_path):
    async def spawn(_): pass
    client=build_client(monkeypatch,tmp_path,spawn)
    response=client.get('/api/jobs/'+'f'*32)
    assert response.status_code==401
    response=client.get('/api/health')
    assert response.json()['ok'] is True


def test_invalid_extension_rejected(monkeypatch,tmp_path):
    async def spawn(_): pass
    client=build_client(monkeypatch,tmp_path,spawn)
    response=client.post('/api/jobs',headers={'X-WorldPilot-Key':'long-test-code-2026'},
        files={'video':('notvideo.txt',b'broken','text/plain')})
    assert response.status_code==400
    assert list(tmp_path.iterdir()) == []


def test_video_upload_and_retrieve(monkeypatch,tmp_path):
    calls=[]
    async def spawn(id): calls.append(id)
    client=build_client(monkeypatch,tmp_path,spawn)
    video=tmp_path/'sample.mp4'
    create_sample_video(video)
    with video.open('rb') as fp:
        response=client.post('/api/jobs',headers={'X-WorldPilot-Key':'long-test-code-2026'},
            files={'video':('sample.mp4',fp,'video/mp4')})
    assert response.status_code==202, response.text
    data=response.json()
    assert data['status']=='queued'
    assert data['frame_count']>=2
    assert calls==[data['id']]
    assert (tmp_path/data['id']/'frames'/'0001.jpg').exists()
    status=client.get('/api/jobs/'+data['id'],headers={'X-WorldPilot-Key':'long-test-code-2026'})
    assert status.json()['status']=='queued'
    forbidden=client.get('/api/jobs/'+data['id']+'/full.ply',headers={'X-WorldPilot-Key':'long-test-code-2026'})
    assert forbidden.status_code==409
    assert client.get('/').status_code==404


def test_path_traversal_blocked(tmp_path):
    from fastapi import HTTPException
    with pytest.raises(HTTPException): get_job_dir(tmp_path,'../../secrets')


def test_completed_ply_can_be_downloaded(monkeypatch, tmp_path):
    async def spawn(_): pass
    client = build_client(monkeypatch, tmp_path, spawn)
    job_id = '0' * 32
    folder = tmp_path / job_id
    folder.mkdir()
    (folder/'manifest.json').write_text(json.dumps({'id':job_id, 'status':'completed', 'frame_count':8}))
    # Tiny synthetic fixture created inside the test, not shipped as a real scene.
    ply = b'ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n'
    (folder/'preview.ply').write_bytes(ply)
    (folder/'full.ply').write_bytes(ply)
    response = client.get('/api/jobs/'+job_id+'/preview.ply',headers={'X-WorldPilot-Key':'long-test-code-2026'})
    assert response.status_code==200 and response.content==ply
    response = client.delete('/api/jobs/'+job_id,headers={'X-WorldPilot-Key':'long-test-code-2026'})
    assert response.status_code==200
    assert not folder.exists()


def test_upload_commits_before_enqueue_with_async_volume(monkeypatch, tmp_path):
    monkeypatch.setenv('WORLD_ACCESS_CODE', 'long-test-code-2026')
    volume = FakeVolume()
    queued = []

    async def enqueue(job_id):
        # The separate GPU container must see committed inputs before it starts.
        volume.commit.aio.assert_awaited_once()
        assert (tmp_path / job_id / 'manifest.json').is_file()
        queued.append(job_id)

    monkeypatch.setattr(service, 'inspect_video', lambda _: {'duration_seconds': 8})
    monkeypatch.setattr(service, 'extract_frames', lambda *_: 3)
    client = TestClient(create_api(volume=volume, enqueue=enqueue, root=tmp_path))
    response = client.post('/api/jobs', headers={'X-WorldPilot-Key': 'long-test-code-2026'},
                           files={'video': ('sample.mp4', b'video-fixture', 'video/mp4')})
    assert response.status_code == 202, response.text
    assert queued == [response.json()['id']]
    volume.reload.aio.assert_awaited_once()
    volume.commit.assert_not_called()
    volume.reload.assert_not_called()


def test_enqueue_failure_commits_cleanup_asynchronously(monkeypatch, tmp_path):
    monkeypatch.setenv('WORLD_ACCESS_CODE', 'long-test-code-2026')
    volume = FakeVolume()

    async def enqueue(_):
        raise RuntimeError('Worker unavailable')

    monkeypatch.setattr(service, 'inspect_video', lambda _: {'duration_seconds': 8})
    monkeypatch.setattr(service, 'extract_frames', lambda *_: 3)
    client = TestClient(create_api(volume=volume, enqueue=enqueue, root=tmp_path))
    with pytest.raises(RuntimeError, match='Worker unavailable'):
        client.post('/api/jobs', headers={'X-WorldPilot-Key': 'long-test-code-2026'},
                    files={'video': ('sample.mp4', b'video-fixture', 'video/mp4')})
    assert list(tmp_path.iterdir()) == []
    assert volume.commit.aio.await_count == 2
    volume.commit.assert_not_called()


def test_health_reports_running_source_revision(monkeypatch, tmp_path):
    async def enqueue(_): pass
    client = build_client(monkeypatch, tmp_path, enqueue)
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['revision'] == service.SERVICE_REVISION
    assert len(response.json()['revision']) == 12


def test_video_validation_error_is_logged_and_cleaned_up(monkeypatch, tmp_path, caplog):
    from fastapi import HTTPException
    monkeypatch.setenv('WORLD_ACCESS_CODE', 'long-test-code-2026')
    volume = FakeVolume()
    enqueue = AsyncMock()

    def reject_video(_):
        raise HTTPException(400, 'Unsupported video resolution.')

    monkeypatch.setattr(service, 'inspect_video', reject_video)
    client = TestClient(create_api(volume=volume, enqueue=enqueue, root=tmp_path))
    with caplog.at_level('WARNING', logger='uvicorn.error'):
        response = client.post('/api/jobs', headers={'X-WorldPilot-Key': 'long-test-code-2026'},
                               files={'video': ('sample.mp4', b'video-fixture', 'video/mp4')})
    assert response.status_code == 400
    assert response.json()['detail'] == 'Unsupported video resolution.'
    assert 'Unsupported video resolution.' in caplog.text
    assert service.SERVICE_REVISION in caplog.text
    assert 'long-test-code-2026' not in caplog.text
    assert list(tmp_path.iterdir()) == []
    volume.reload.aio.assert_awaited_once()
    volume.commit.aio.assert_awaited_once()
    enqueue.assert_not_awaited()
