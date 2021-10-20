import os
import ast

from hub.constants import ENV_DEV_MODE, ENV_LOCAL_MODE

TOKEN_FILE_PATH = os.path.expanduser("~/.activeloop/token")
REPORTING_CONFIG_FILE_PATH = os.path.expanduser("~/.activeloop/reporting_config.json")

HUB_REST_ENDPOINT = "https://app.activeloop.ai"
HUB_REST_ENDPOINT_DEV = "https://app.dev.activeloop.ai"
HUB_REST_ENDPOINT_LOCAL = "http://localhost:5000"

def set_dev(value):
    os.environ[ENV_DEV_MODE] = str(value)
def is_dev() -> bool:
    return ast.literal_eval(os.environ.get(ENV_DEV_MODE, "False"))
def is_local() -> bool:
    return ast.literal_eval(os.environ.get(ENV_LOCAL_MODE, "False"))


GET_TOKEN_SUFFIX = "/api/user/token"
REGISTER_USER_SUFFIX = "/api/user/register"
GET_DATASET_CREDENTIALS_SUFFIX = "/api/org/{}/ds/{}/creds"
CREATE_DATASET_SUFFIX = "/api/dataset/create"
DATASET_SUFFIX = "/api/dataset"
UPDATE_SUFFIX = "/api/org/{}/dataset/{}"
LIST_DATASETS = "/api/datasets/{}"
GET_USER_PROFILE = "/api/user/profile"

DEFAULT_REQUEST_TIMEOUT = 170

HUB_AUTH_TOKEN = "HUB_AUTH_TOKEN"
