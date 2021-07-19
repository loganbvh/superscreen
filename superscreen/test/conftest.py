import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def set_ray_redis_retries():
    old_environ = dict(os.environ)
    os.environ["RAY_START_REDIS_WAIT_RETRIES"] = "48"
    yield
    # Will be executed after the last test
    os.environ.clear()
    os.environ.update(old_environ)
