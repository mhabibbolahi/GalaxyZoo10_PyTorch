import shutil
import uvicorn
import numpy as np
from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import cnn_use_model
app = FastAPI()

templates = Jinja2Templates(directory="templates")


class Item(BaseModel):
    part: str
    sections: int
    scenario: str


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload-file/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    file_location = f"static/{file.filename}"
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = cnn_use_model.main(file_location,'model.pth')

    return templates.TemplateResponse("image_result.html",
                                      {"request": request, "file_url": file_location, "network": 'DeepCNN-5Block',"result":result})


app.mount("/static", StaticFiles(directory="static"), name="static")
if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)
