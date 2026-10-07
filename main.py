import os
import logging
import pandas as pd
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "model/dissolved_oxygen_model.pkl"
)

DO_LOW_THRESHOLD = float(
    os.getenv("DO_LOW_THRESHOLD", "5.0")
)

DO_RECOVERY_THRESHOLD = float(
    os.getenv("DO_RECOVERY_THRESHOLD", "6.0")
)


# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# Load ML model
# --------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Dissolved oxygen model loaded successfully")

except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Aquaculture Water Monitoring API",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Five main variables
# --------------------------------------------------

temperature = 0.0
turbidity = 0.0
dissolved_oxygen = 0.0

mode = "AUTO"
aerator_state = "OFF"


# --------------------------------------------------
# Request models
# --------------------------------------------------

class SensorData(BaseModel):
    temperature: float
    turbidity: float


class ModeRequest(BaseModel):
    mode: str


class AeratorRequest(BaseModel):
    state: str


# --------------------------------------------------
# ML prediction
# --------------------------------------------------

def predict_dissolved_oxygen(
    water_temperature: float,
    water_turbidity: float
) -> float:

    input_data = pd.DataFrame([
        {
            "water_temperature": water_temperature,
            "turbidity": water_turbidity
        }
    ])

    prediction = model.predict(input_data)

    return float(prediction[0])


# --------------------------------------------------
# Automatic aerator control
# --------------------------------------------------

def update_auto_aerator():

    global aerator_state

    # Only AUTO mode controls the aerator
    if mode != "AUTO":
        return

    # DO too low -> turn aerator ON
    if dissolved_oxygen < DO_LOW_THRESHOLD:

        if aerator_state != "ON":
            aerator_state = "ON"

            logger.info(
                f"AUTO: DO={dissolved_oxygen:.2f} "
                f"< {DO_LOW_THRESHOLD} -> Aerator ON"
            )

    # DO recovered -> turn aerator OFF
    elif dissolved_oxygen > DO_RECOVERY_THRESHOLD:

        if aerator_state != "OFF":
            aerator_state = "OFF"

            logger.info(
                f"AUTO: DO={dissolved_oxygen:.2f} "
                f"> {DO_RECOVERY_THRESHOLD} -> Aerator OFF"
            )

    # Between the thresholds:
    # keep the previous aerator state


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Aquaculture Water Monitoring API",
        "status": "running"
    }


# --------------------------------------------------
# ESP32 -> Backend
# Receive temperature and turbidity
# --------------------------------------------------

@app.post("/api/sensors")
def receive_sensor_data(data: SensorData):

    global temperature
    global turbidity
    global dissolved_oxygen

    # Update sensor variables
    temperature = data.temperature
    turbidity = data.turbidity

    # Predict dissolved oxygen
    dissolved_oxygen = predict_dissolved_oxygen(
        temperature,
        turbidity
    )

    # If AUTO mode, update aerator
    update_auto_aerator()

    logger.info(
        f"Sensors | Temperature={temperature:.2f} "
        f"| Turbidity={turbidity:.2f} "
        f"| Predicted DO={dissolved_oxygen:.2f} "
        f"| Mode={mode} "
        f"| Aerator={aerator_state}"
    )

    return {
        "temperature": temperature,
        "turbidity": turbidity,
        "dissolved_oxygen": dissolved_oxygen,
        "mode": mode,
        "aerator_state": aerator_state
    }


# --------------------------------------------------
# Frontend -> Backend
# Get current sensor/ML/aerator data
# --------------------------------------------------

@app.get("/api/data")
def get_data():

    return {
        "temperature": temperature,
        "turbidity": turbidity,
        "dissolved_oxygen": dissolved_oxygen,
        "mode": mode,
        "aerator_state": aerator_state
    }


# --------------------------------------------------
# ESP32 -> Backend
# ESP32 constantly polls this endpoint
# --------------------------------------------------

@app.get("/api/aerator/state")
def get_aerator_state():

    return {
        "mode": mode,
        "state": aerator_state
    }


# --------------------------------------------------
# Frontend -> Backend
# Change AUTO / MANUAL mode
# --------------------------------------------------

@app.post("/api/aerator/mode")
def change_mode(data: ModeRequest):

    global mode
    global aerator_state

    requested_mode = data.mode.upper()

    if requested_mode not in ["AUTO", "MANUAL"]:
        raise HTTPException(
            status_code=400,
            detail="Mode must be AUTO or MANUAL"
        )

    mode = requested_mode

    logger.info(f"Mode changed to {mode}")

    # When switching back to AUTO,
    # immediately evaluate the current DO.
    if mode == "AUTO":
        update_auto_aerator()

    return {
        "mode": mode,
        "state": aerator_state
    }


# --------------------------------------------------
# Frontend -> Backend
# Manual aerator control
# --------------------------------------------------

@app.post("/api/aerator/manual")
def manual_aerator_control(data: AeratorRequest):

    global aerator_state

    requested_state = data.state.upper()

    if mode != "MANUAL":
        raise HTTPException(
            status_code=409,
            detail="Aerator can only be manually controlled in MANUAL mode"
        )

    if requested_state not in ["ON", "OFF"]:
        raise HTTPException(
            status_code=400,
            detail="State must be ON or OFF"
        )

    aerator_state = requested_state

    logger.info(
        f"MANUAL: Aerator changed to {aerator_state}"
    )

    return {
        "mode": mode,
        "state": aerator_state
    }