from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request, BackgroundTasks


app = FastAPI(
    title = "🩺Diabetes AutoML",
    description = "Custom AutoML Engine for Diabetes Detection.",
    version = "1.0.0",
)

app.mount("/static", 
          StaticFiles(directory="static"), 
          name="static")


templates = Jinja2Templates(directory="templates")

automl_state = {
    "status": "idle",
    "progress": 0,
    "completed": 0,
    "current_model": None,
    "total": 0,
    "results": [],
    "error": None
}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
         request=request,
        name="index.html"
    )

@app.get("/api/status")
def status():
    return {"status": "running",
            "message": "Diabetes Detection Automl Engine is running."
            } 

@app.get("/api/models")
def get_models():

    models = [
        "LogisticRegression",
        "RandomForestClassifier",
        "DecisionTreeClassifier"
    ]

    return {"models": models}

@app.get("/api/automl/status")
def get_automl_status():
    return automl_state


@app.post("/api/automl/run")
def run_automl(background_tasks: BackgroundTasks):

    if automl_state["status"] == "running":
        return {
            "status" : "already_running",
            "message": "Automl is already running."}

    automl_state["status"] = "starting"
    automl_state["progress"] = 0
    automl_state["completed"] = 0
    automl_state["current_model"] = None
    automl_state["total"] = 0
    automl_state["results"] = []
    automl_state["error"] = None

    background_tasks.add_task(run_automl_background)
    return {
        "status": "started",
        "message": "Automl process started."}

def run_automl_background():

    global automl_state

    try:

        automl_state["status"] = "running"

        import automl

        selected_models = automl.models_to_test
        search_spaces = automl.custom_search_spaces

        def progress_callback(
                completed, 
                total, 
                model_name, 
                params, 
        ):
            automl_state["completed"] = completed
            automl_state["total"] = total
            automl_state["current_model"] = model_name


            if total > 0:
                automl_state["progress"] = round(
                    (completed / total) * 100, 1
                )

        leaderboard = automl.run_automl(
            selected_models=selected_models,
            search_spaces=search_spaces,
            progress_callback=progress_callback
        )

        automl_state["results"] = leaderboard

        if leaderboard:
            best = leaderboard[0]
            automl_state["best_model"] = best.get("model")
            automl_state["f1"] = best.get("f1")
            automl_state["roc_auc"] = best.get("roc_auc")
            automl_state["precision"] = best.get("precision")
            automl_state["recall"] = best.get("recall")
            automl_state["accuracy"] = best.get("accuracy")

        automl_state["progress"] = 100
        automl_state["status"] = "completed"

    except Exception as e:

                automl_state["status"] = "error"
                automl_state["error"] = str(e)



        