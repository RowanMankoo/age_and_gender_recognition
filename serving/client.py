import httpx
from google.auth.transport.requests import Request
from google.oauth2 import service_account


class AgeAndGenderClient:
    def __init__(self, url, service_account_file=None):
        self.url = url
        self.id_token = None
        if service_account_file:
            self.id_token = self.get_id_token(service_account_file, url)

    def get_id_token(self, service_account_file, target_audience):
        credentials = service_account.IDTokenCredentials.from_service_account_file(
            service_account_file,
            target_audience=target_audience,
        )

        request = Request()
        credentials.refresh(request)
        return credentials.token

    async def send_image(self, image_path):
        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            headers = {}
            if self.id_token:
                headers["Authorization"] = f"Bearer {self.id_token}"
            async with httpx.AsyncClient() as client:
                response = await client.post(self.url, files=files, headers=headers)

        return response.json()
