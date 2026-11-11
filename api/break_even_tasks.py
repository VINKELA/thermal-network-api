"""Durable Celery entry point, isolated from unrelated network tasks."""
from celery import shared_task
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError


@shared_task(bind=True, name='api.break_even_tasks.run_break_even_study',
             acks_late=True, reject_on_worker_lost=True, track_started=True,
             autoretry_for=(OSError,), retry_backoff=True, retry_backoff_max=30,
             retry_kwargs={'max_retries': 3})
def run_break_even_study(self, run_id, owner_id, data):
    from .break_even_views import execute
    from .access_control import assert_can_view_project
    from django.contrib.auth import get_user_model
    from rest_framework.exceptions import NotFound

    def report(message):
        # Eager-mode tests do not require a running result backend.
        if not self.request.is_eager:
            try:
                self.update_state(state='PROGRESS', meta={'run_id': run_id, 'message': message})
            except (OperationalError, RedisError, OSError):
                # File snapshots remain authoritative if Redis briefly disconnects.
                pass

    # A queued job may begin after project membership has been revoked.
    user = get_user_model().objects.filter(pk=owner_id).first()
    if user is None:
        raise NotFound('Study owner no longer exists.')
    assert_can_view_project(user, data['project_id'])
    return execute(run_id, owner_id, data, progress_callback=report)
