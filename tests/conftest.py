import pytest


@pytest.fixture(scope="session")
def spark():
    from demand_lakehouse.spark import get_spark

    s = get_spark("local[2]", partitions=4, memory="1g")
    yield s
    s.stop()
