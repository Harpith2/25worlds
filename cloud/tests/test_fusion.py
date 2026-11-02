import asyncio
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from service import create_api
import spatial


def patch_points(start=0., end=1., shift=0):
    x, y = np.meshgrid(np.linspace(start, end, 28), np.linspace(0, 1, 28))
    return np.stack([x.ravel()+shift, y.ravel(), np.zeros(x.size)], axis=-1)


def obj(points, oid=0):
    return {"object_id": oid, "label": "chair", "sample_xyz": points.tolist(), "grounded_points": len(points)}


def frame(i, objects):
    return {"frame_index": i, "object_concept": "chair", "objects": objects, "floor": None}


def test_partial_views_merge_without_changing_shared_coordinates():
    a = patch_points(0, .75)
    b = patch_points(.25, 1.)
    result = spatial.fuse_observations([frame(0, [obj(a)]), frame(2, [obj(b)])], scene_scale=5.)
    assert len(result['objects']) == 1
    fused = result['objects'][0]
    assert fused['frame_indices'] == [0, 2]
    assert fused['observation_count'] == 2
    assert fused['bbox_min_xyz'][0] == 0
    assert fused['bbox_max_xyz'][0] == 1
    assert fused['unique_sample_points'] > len(a)
    assert max(p[2] for p in fused['sample_xyz']) == 0


def test_nearby_disjoint_objects_remain_separate():
    result = spatial.fuse_observations([frame(0, [obj(patch_points())]), frame(1, [obj(patch_points(shift=1.3))])], scene_scale=5.)
    assert len(result['objects']) == 2


def test_same_frame_detections_never_merge():
    a = patch_points()
    result = spatial.fuse_observations([frame(0, [obj(a, 0), obj(a, 1)])], scene_scale=5.)
    assert len(result['objects']) == 2


def test_ambiguous_match_is_not_assigned_to_either_neighbor():
    a = patch_points()
    result = spatial.fuse_observations([frame(0, [obj(a, 0), obj(a, 1)]), frame(1, [obj(a)])], scene_scale=5.)
    assert len(result['objects']) == 3
    assert any('ambiguous' in w for w in result['warnings'])


def test_repeated_samples_deduplicate_and_ids_are_deterministic():
    a = patch_points()
    frames = [frame(2, [obj(a)]), frame(0, [obj(a)])]
    result = spatial.fuse_observations(frames, scene_scale=5.)
    again = spatial.fuse_observations(frames[::-1], scene_scale=5.)
    assert result == again
    assert result['objects'][0]['unique_sample_points'] == len(a)
    assert result['objects'][0]['observation_count'] == 2


def test_empty_results_and_invalid_scale():
    assert spatial.fuse_observations([frame(0, []), frame(1, [])], scene_scale=5.)['objects'] == []
    with pytest.raises(ValueError):
        spatial.fuse_observations([], scene_scale=0)


def test_floor_only_analysis_calls_sam_once(monkeypatch):
    y, x = np.mgrid[:20, :20]
    points = np.stack([x/20, np.zeros_like(x), y/20], -1)
    mask = spatial.SAMInstance(0, 'floor', np.ones((20, 20), bool), (0, 0, 20, 20), .9)
    segment = AsyncMock(return_value=[mask])
    monkeypatch.setattr(spatial, 'segment_image', segment)
    result = asyncio.run(spatial.analyze_frame(np.zeros((20, 20, 3), np.uint8), points, np.ones((20,20),bool), object_concept='floor'))
    segment.assert_awaited_once()
    assert result['floor'] is not None


def client_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv('WORLD_ACCESS_CODE','fusion-test-code')
    folder = tmp_path / ('b'*32)
    folder.mkdir()
    (folder/'manifest.json').write_text(json.dumps({'status':'completed','spatial_ready':True,'spatial_frames':6}))
    volume=SimpleNamespace(commit=SimpleNamespace(aio=AsyncMock()),reload=SimpleNamespace(aio=AsyncMock()))
    spawn=AsyncMock(return_value=SimpleNamespace(object_id=None))
    return TestClient(create_api(volume=volume,enqueue=AsyncMock(),enqueue_analysis=AsyncMock(),enqueue_fusion=spawn,root=tmp_path)),spawn,folder


def test_fusion_endpoint_queues_deduplicates_and_polls_saved_result(tmp_path, monkeypatch):
    client, spawn, folder = client_fixture(tmp_path,monkeypatch)
    url='/api/jobs/'+'b'*32+'/fusion'
    headers={'x-worldpilot-key':'fusion-test-code'}
    args={'concept':'Chair','frame_indices':[5,0,2]}
    assert client.post(url,headers=headers,json=args).json()['status']=='queued'
    spawn.assert_awaited_once_with('b'*32,'chair',[0,2,5])
    assert client.post(url,headers=headers,json=args).json()['status']=='queued'
    assert spawn.await_count==1
    digest=hashlib.sha256(json.dumps({'concept':'chair','frame_indices':[0,2,5]},sort_keys=True).encode()).hexdigest()[:12]
    target=folder/'analysis'/f'fusion-{digest}.json'
    target.write_text(json.dumps({'objects':[], 'frame_indices':[0,2,5]}))
    response=client.get(url+'?concept=chair&frame_indices=0,2,5',headers=headers)
    assert response.json()['status']=='completed'
    assert response.json()['result']['frame_indices']==[0,2,5]
    assert client.post(url,headers=headers,json=args).json()['status']=='completed'
    assert spawn.await_count==1


@pytest.mark.parametrize('frames',[[0],[0,0],[True,1],[0,9],[-1,2],list(range(9)),None])
def test_fusion_endpoint_rejects_invalid_frames(tmp_path,monkeypatch,frames):
    client,spawn,_=client_fixture(tmp_path,monkeypatch)
    response=client.post('/api/jobs/'+'b'*32+'/fusion',headers={'x-worldpilot-key':'fusion-test-code'},json={'concept':'chair','frame_indices':frames})
    assert response.status_code==400
    spawn.assert_not_awaited()


def test_fusion_enqueue_failure_reports_retryable_error(tmp_path,monkeypatch):
    client,spawn,_=client_fixture(tmp_path,monkeypatch)
    spawn.side_effect=RuntimeError('offline')
    response=client.post('/api/jobs/'+'b'*32+'/fusion',headers={'x-worldpilot-key':'fusion-test-code'},json={'concept':'chair','frame_indices':[0,2]})
    assert response.status_code==503
