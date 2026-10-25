from pydantic import BaseModel,Field
from typing import List,Optional,Literal

class  Review(BaseModel):
    Product: str ='No Info'  #if nothing is provided, it will be an empty string
    pos: List[str]
    nav: List[str]
    Reviewer: Optional[str] = None   #if name is not provided, it wont be issue
    rating: float=Field(ge=0, le=5, description="Rating should be between 0 and 5") 
