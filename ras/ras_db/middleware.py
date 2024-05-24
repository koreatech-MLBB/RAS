# middleware.py

import re
from django.utils.deprecation import MiddlewareMixin


class AuthorizationHeaderMid(MiddlewareMixin):
    def process_request(self, request):
        pattern = re.compile(r'^/.+/running(/(main|info(/(\d+))?)?)?$')
        if pattern.match(request.path) or request.path.endswith('profile'):
            token = request.META.get('HTTP_AUTHORIZATION', None)

            if not token:
                auth_token = (request.COOKIES.get('authToken', None)
                              or request.session.get('authToken', None)
                              or request.headers.get('Authorization', None))
                if auth_token:
                    request.META['HTTP_AUTHORIZATION'] = f'Token {auth_token}'
