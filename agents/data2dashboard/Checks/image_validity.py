from PIL import Image

image_path = "~/Desktop/Data_Viz_agent/data-dashboard/results/apm_application_metrics/Create a time series graph for the call rate received by the application with respect to time./call_rate_over_time_0.png"

try:
    with Image.open(image_path) as img:
        img.verify()  # Will raise if image is invalid
        print("image verified")
except Exception as e:
    raise ValueError(f"Image file is invalid or corrupted: {e}")
