from django.utils.translation import gettext_lazy as _
from rest_framework import exceptions
from rest_framework.authentication import get_authorization_header
from rest_framework.authtoken.models import Token


class BearerOrTokenAuthentication:
    """
    Accept both DRF's standard "Token <key>" header and the older frontend
    convention of sending "Bearer <key>".
    """

    model = Token
    supported_keywords = {"token", "bearer"}

    def authenticate(self, request):
        auth = get_authorization_header(request).split()
        if not auth:
            return None

        keyword = auth[0].decode("utf-8").lower()
        if keyword not in self.supported_keywords:
            return None

        if len(auth) == 1:
            raise exceptions.AuthenticationFailed(_("Invalid token header. No credentials provided."))
        if len(auth) > 2:
            raise exceptions.AuthenticationFailed(_("Invalid token header. Token string should not contain spaces."))

        try:
            token = auth[1].decode("utf-8")
        except UnicodeError:
            raise exceptions.AuthenticationFailed(_("Invalid token header. Token string should be UTF-8."))

        return self.authenticate_credentials(token)

    def authenticate_credentials(self, key):
        try:
            token = self.model.objects.select_related("user").get(key=key)
        except self.model.DoesNotExist:
            raise exceptions.AuthenticationFailed(_("Invalid token."))

        if not token.user.is_active:
            raise exceptions.AuthenticationFailed(_("User inactive or deleted."))

        return token.user, token

    def authenticate_header(self, request):
        return "Token"
