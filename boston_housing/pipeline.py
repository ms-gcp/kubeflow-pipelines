import kfp
from kfp import dsl

import re
import requests
from urllib.parse import urlsplit

def get_istio_auth_session(url: str, username: str, password: str) -> dict:
    # define the default return object
    auth_session = {
        "endpoint_url": url,    # KF endpoint URL
        "redirect_url": None,   # KF redirect URL, if applicable
        "dex_login_url": None,  # Dex login URL (for POST of credentials)
        "is_secured": None,     # True if KF endpoint is secured
        "session_cookie": None  # Resulting session cookies in the form "key1=value1; key2=value2"
    }

    # use a persistent session (for cookies)
    with requests.Session() as s:

        ################
        # Determine if Endpoint is Secured
        ################
        resp = s.get(url, allow_redirects=True)
        if resp.status_code != 200:
            raise RuntimeError(
                f"HTTP status code '{resp.status_code}' for GET against: {url}"
            )

        auth_session["redirect_url"] = resp.url

        # if we were NOT redirected, then the endpoint is UNSECURED
        if len(resp.history) == 0:
            auth_session["is_secured"] = False
            return auth_session
        else:
            auth_session["is_secured"] = True

        ################
        # Get Dex Login URL
        ################
        redirect_url_obj = urlsplit(auth_session["redirect_url"])

        # if we are at `/auth?=xxxx` path, we need to select an auth type
        if re.search(r"/auth$", redirect_url_obj.path): 
            
            #######
            # TIP: choose the default auth type by including ONE of the following
            #######
            
            # OPTION 1: set "staticPasswords" as default auth type
            redirect_url_obj = redirect_url_obj._replace(
                path=re.sub(r"/auth$", "/auth/local", redirect_url_obj.path)
            )
            # OPTION 2: set "ldap" as default auth type 
            # redirect_url_obj = redirect_url_obj._replace(
            #     path=re.sub(r"/auth$", "/auth/ldap", redirect_url_obj.path)
            # )
            
        # if we are at `/auth/xxxx/login` path, then no further action is needed (we can use it for login POST)
        if re.search(r"/auth/.*/login$", redirect_url_obj.path):
            auth_session["dex_login_url"] = redirect_url_obj.geturl()

        # else, we need to be redirected to the actual login page
        else:
            # this GET should redirect us to the `/auth/xxxx/login` path
            resp = s.get(redirect_url_obj.geturl(), allow_redirects=True)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP status code '{resp.status_code}' for GET against: {redirect_url_obj.geturl()}"
                )

            # set the login url
            auth_session["dex_login_url"] = resp.url

        ################
        # Attempt Dex Login
        ################
        resp = s.post(
            auth_session["dex_login_url"],
            data={"login": username, "password": password},
            allow_redirects=True
        )
        if len(resp.history) == 0:
            raise RuntimeError(
                f"Login credentials were probably invalid - "
                f"No redirect after POST to: {auth_session['dex_login_url']}"
            )

        # store the session cookies in a "key1=value1; key2=value2" string
        auth_session["session_cookie"] = "; ".join([f"{c.name}={c.value}" for c in s.cookies])

    return auth_session

KUBEFLOW_ENDPOINT = "http://localhost:8080"
KUBEFLOW_USERNAME = "user@example.com"
KUBEFLOW_PASSWORD = "12341234"

auth_session = get_istio_auth_session(
    url=KUBEFLOW_ENDPOINT,
    username=KUBEFLOW_USERNAME,
    password=KUBEFLOW_PASSWORD
)

def preprocess_op():

    return dsl.ContainerOp(
        name='Preprocess Data',
        image='sayyeddocker/boston_pipeline_preprocessing:latest',
        arguments=[],
        file_outputs={
            'x_train': '/app/x_train.npy',
            'x_test': '/app/x_test.npy',
            'y_train': '/app/y_train.npy',
            'y_test': '/app/y_test.npy',
        }
    )

def train_op(x_train, y_train):

    return dsl.ContainerOp(
        name='Train Model',
        image='sayyeddocker/boston_pipeline_train:latest',
        arguments=[
            '--x_train', x_train,
            '--y_train', y_train
        ],
        file_outputs={
            'model': '/app/model.pkl'
        }
    )

def test_op(x_test, y_test, model):

    return dsl.ContainerOp(
        name='Test Model',
        image='sayyeddocker/boston_pipeline_test:latest',
        arguments=[
            '--x_test', x_test,
            '--y_test', y_test,
            '--model', model
        ],
        file_outputs={
            'mean_squared_error': '/app/output.txt'
        }
    )

def deploy_model_op(model):

    return dsl.ContainerOp(
        name='Deploy Model',
        image='sayyeddocker/boston_pipeline_deploy_model:latest',
        arguments=[
            '--model', model
        ]
    )

@dsl.pipeline(
   name='Boston Housing Pipeline',
   description='An example pipeline that trains and logs a regression model.'
)
def boston_pipeline():
    _preprocess_op = preprocess_op()
    
    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_train']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_train'])
    ).after(_preprocess_op)

    _test_op = test_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['x_test']),
        dsl.InputArgumentPath(_preprocess_op.outputs['y_test']),
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_train_op)

    deploy_model_op(
        dsl.InputArgumentPath(_train_op.outputs['model'])
    ).after(_test_op)

client = kfp.Client(host=f"{KUBEFLOW_ENDPOINT}/pipeline", cookies=auth_session["session_cookie"])
client.create_run_from_pipeline_func(boston_pipeline, arguments={})
print(client.list_experiments())