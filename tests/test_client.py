import pytest

from serving.client import AgeAndGenderClient


@pytest.mark.asyncio
async def test_cloud_run_endpoint():

    url = "https://age-and-gender-apis-fij2usbs6a-nw.a.run.app/predict"
    key_path = "rowan-420019-d016700a138a.json"
    image_path = "Data/crop_part1/1_0_0_20161219140623097.jpg.chip.jpg"

    client = AgeAndGenderClient(url, key_path)
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    resp = await client.send_image(image_data)

    assert resp["age"] > 0
    assert resp["gender"] in ["Male", "Female"]


@pytest.mark.asyncio
async def test_local_endpoint():

    url = "http://localhost:8080/predict"
    image_path = "Data/crop_part1/1_0_0_20161219140623097.jpg.chip.jpg"

    client = AgeAndGenderClient(url)
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    resp = await client.send_image(image_data)

    assert resp["age"] > 0
    assert resp["gender"] in ["Male", "Female"]
