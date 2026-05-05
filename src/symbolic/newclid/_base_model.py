from pydantic import BaseModel, ConfigDict


class NewclidModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
