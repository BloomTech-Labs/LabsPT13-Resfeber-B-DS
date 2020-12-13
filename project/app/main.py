from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api import predict, viz

app = FastAPI(
    title='Resfeber B DS API',
    description='An api for serving up gas cost predictions, arbnb price price predictions, and visulalizations for both.',
    version='0.1',
    docs_url='/',
)

app.include_router(predict.router)
app.include_router(viz.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

if __name__ == '__main__':
    uvicorn.run(app)
