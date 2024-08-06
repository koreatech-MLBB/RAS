# middleware.py

import re
from django.utils.deprecation import MiddlewareMixin


class AuthorizationHeaderMid(MiddlewareMixin):
    def process_request(self, request):
        # print("Middleware")
        # print(request.META)
        pattern = re.compile(r'^/.+/running(/(main|info(/(\d+))?)?)?$')
        if (pattern.match(request.path) or request.path.endswith('profile') or 'ming' in request.path
                or 'save' in request.path or 'in' in request.path):
            token = request.META.get('HTTP_AUTHORIZATION', None)
            token = token.split(',')[0] if token is not None else token
            # print("token", token)
            if not token:
                auth_token = (request.COOKIES.get('authToken', None)
                              or request.session.get('authToken', None)
                              or request.headers.get('Authorization', None))
                # print("auth: ", auth_token)

                if auth_token:
                    request.META['HTTP_AUTHORIZATION'] = f'Token {auth_token}'
