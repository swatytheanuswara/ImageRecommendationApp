"""This module is to create standard response of the API"""
from apps import constant
from fastapi.responses import JSONResponse

class StandardResponse:
    """This class is universal to return standard API responses

    Attributes:
        status (int): The http status response from API
        data (dict/list): The Data from API
        message (str): The message from the API
    """

    def __init__(self, success: str, status: int, message: str, data: dict) -> None:
        """This function defines arguments that are used in the class

        Arguments:
            status (int): The http status response from API
            data (dict/list): The Data from API
            message (str): The message from the API

        Returns:
            Returns the API standard response
        """
        self.success = success
        self.status = status
        self.message = message
        self.data = data

    @property
    def make(self) -> dict:
        self.status = constant.STATUS_SUCCESS if self.status in [201, 200] else constant.STATUS_FAIL
        response = {"success": self.message,'status': self.status_code, 'data': self.data, 'message': self.msg}
        return JSONResponse(content=response,status_code=self.status_code)