"""Local background study runner with user-scoped, atomic on-disk results.

Avoids tying multi-year weather retrieval to an HTTP request. This is a bounded
local runner, not a durable distributed queue; stale interrupted jobs can be rerun.
"""
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from django.conf import settings
from django.db import close_old_connections
from rest_framework import serializers
from rest_framework.response import Response
from rest_framework.views import APIView
from .access_control import assert_can_view_project
from .intelligence.heating_sensitivity import SensitivityRequest, prepare_sensitivity_case, run_pipe_sensitivity

logger=logging.getLogger(__name__)
_slots=threading.BoundedSemaphore(2)


def study_path(run_id):
    return Path(settings.BASE_DIR)/'data'/'sensitivity_runs'/str(uuid.UUID(str(run_id)))


def write_study_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(data,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def execute_study(run_id, owner, options, case):
    close_old_connections()
    folder=study_path(run_id)
    state=dict(run_id=run_id,user_id=owner,project_id=options['project_id'],status='running',message='Starting study',updated_at=time.time())
    def progress(message):
        state.update(message=message,updated_at=time.time())
        write_study_json(folder/'status.json',state)
    try:
        progress('Validating study inputs')
        result=run_pipe_sensitivity(options,case,progress)
        write_study_json(folder/'result.json',result)
        state.update(status='complete',message='Study complete',updated_at=time.time())
    except Exception as exc:
        logger.exception('Pipe sensitivity study %s failed',run_id)
        state.update(status='failed',message=str(exc) if isinstance(exc,ValueError) else 'The study could not finish. Check weather connectivity and the server log, then retry.',updated_at=time.time())
    finally:
        try:
            write_study_json(folder/'status.json',state)
        finally:
            close_old_connections()
            _slots.release()


class PipeSensitivityStartView(APIView):
    def post(self, request):
        validation=SensitivityRequest(data=request.data)
        validation.is_valid(raise_exception=True)
        options=validation.validated_data
        assert_can_view_project(request.user,options['project_id'])
        try:
            case=prepare_sensitivity_case(options)
        except ValueError as exc:
            raise serializers.ValidationError({'detail':str(exc)}) from exc
        if not _slots.acquire(blocking=False):
            return Response({'detail':'Two studies are already running. Wait for completion and retry.'},status=429)
        run_id=str(uuid.uuid4())
        try:
            write_study_json(study_path(run_id)/'status.json',dict(run_id=run_id,user_id=request.user.id,project_id=options['project_id'],status='running',message='Queued locally',updated_at=time.time()))
            threading.Thread(target=execute_study,args=(run_id,request.user.id,options,case),daemon=True,name='pipe-sensitivity-'+run_id).start()
        except Exception:
            _slots.release()
            raise
        return Response({'run_id':run_id,'status':'running'},status=202)


class PipeSensitivityStatusView(APIView):
    def get(self, request, run_id):
        folder=study_path(run_id)
        if not (folder/'status.json').exists():
            return Response({'detail':'Study not found.'},status=404)
        state=json.loads((folder/'status.json').read_text(encoding='utf-8'))
        if state['user_id']!=request.user.id:
            return Response({'detail':'Study not found.'},status=404)
        assert_can_view_project(request.user,state['project_id'])
        state.pop('user_id')
        if state['status']=='running' and time.time()-state['updated_at']>900:
            state.update(status='interrupted',message='This local run stopped reporting progress, possibly after an API restart. Start a new study.')
        if state['status']=='complete':
            state['result']=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        return Response(state)
