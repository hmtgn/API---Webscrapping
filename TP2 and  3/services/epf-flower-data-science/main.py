from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.app import get_application
from src.api.routes import data
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

app = get_application()


# Redirect root endpoint to Swagger documentation
@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# Custom error handler for 404
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "The requested resource was not found on this server."}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True, port=8080)

