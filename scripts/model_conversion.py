import xgboost
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Load your trained .bst model
model = xgboost.Booster()
model.load_model("models/xgb_geolocation_fraud_model_updated.bst")
print(model.feature_names)

feature_names = [f"f{i}" for i in range(20)]  # assuming 20 features
model.feature_names = feature_names
# Define input shape
initial_type = [("input", FloatTensorType([1, 20]))]  # 20 features, batch size 1

# Convert to ONNX model
onnx_model = convert_xgboost(model, initial_types=initial_type)

# Save to file
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
