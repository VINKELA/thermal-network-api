from django.db.models import Q
from rest_framework.exceptions import NotFound, PermissionDenied

from .models import ProjectMembership, _project_id_for_instance


ROLE_RANK = {
    ProjectMembership.Role.VIEWER: 1,
    ProjectMembership.Role.EDITOR: 2,
    ProjectMembership.Role.OWNER: 3,
}


def _is_authenticated(user):
    return bool(user and user.is_authenticated)


def membership_queryset(user):
    if not _is_authenticated(user):
        return ProjectMembership.objects.none()
    return ProjectMembership.objects.filter(user=user)


def visible_project_ids(user):
    if not _is_authenticated(user):
        return []
    if user.is_superuser:
        return None
    return list(membership_queryset(user).values_list("project_id", flat=True))


def project_access_q(user, *project_lookups):
    ids = visible_project_ids(user)
    if ids is None:
        return Q()
    if not ids:
        return Q(pk__in=[])

    query = Q()
    for lookup in project_lookups:
        if lookup:
            query |= Q(**{f"{lookup}__in": ids})
    return query if query.children else Q(pk__in=[])


def apply_project_access(queryset, user, project_lookup):
    ids = visible_project_ids(user)
    if ids is None:
        return queryset
    if not ids:
        return queryset.none()
    return queryset.filter(**{f"{project_lookup}__in": ids})


def project_role_for_user(user, project_id):
    if not _is_authenticated(user) or project_id is None:
        return None
    if user.is_superuser:
        return ProjectMembership.Role.OWNER
    membership = membership_queryset(user).filter(project_id=project_id).first()
    return membership.role if membership else None


def user_can_view_project(user, project_id):
    return project_role_for_user(user, project_id) is not None


def user_can_edit_project(user, project_id):
    role = project_role_for_user(user, project_id)
    return role in {ProjectMembership.Role.OWNER, ProjectMembership.Role.EDITOR}


def user_can_manage_project_users(user, project_id):
    return user_can_edit_project(user, project_id)


def can_grant_role(granting_user, project_id, requested_role):
    grantor_role = project_role_for_user(granting_user, project_id)
    if grantor_role not in {ProjectMembership.Role.OWNER, ProjectMembership.Role.EDITOR}:
        return False
    return ROLE_RANK.get(requested_role, 0) <= ROLE_RANK.get(grantor_role, 0)


def assert_can_view_project(user, project_id):
    if not user_can_view_project(user, project_id):
        raise NotFound("Project not found.")


def assert_can_edit_project(user, project_id):
    if not user_can_edit_project(user, project_id):
        raise PermissionDenied("You do not have edit access to this project.")


def assert_can_manage_project_users(user, project_id):
    if not user_can_manage_project_users(user, project_id):
        raise PermissionDenied("You do not have access to manage users for this project.")


def assert_can_grant_role(user, project_id, requested_role):
    if not can_grant_role(user, project_id, requested_role):
        raise PermissionDenied("You cannot grant a role higher than your own project role.")


def assert_can_view_instance(user, instance):
    project_id = _project_id_for_instance(instance)
    assert_can_view_project(user, project_id)


def assert_can_edit_instance(user, instance):
    project_id = _project_id_for_instance(instance)
    assert_can_edit_project(user, project_id)
