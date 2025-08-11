import coremltools as ct
import torch

# Assume 'model' is your trained PyTorch model
# and 'example_input' is an example input tensor with the correct shape and data type
# Call model.eval() before exporting
model.eval()

mlmodel = ct.convert(
    model,
    inputs=[ct.TensorType(shape=yolov8n.pt)], # Define input shape
    convert_to="mlprogram", # Recommended for newer Core ML versions
    source="pytorch"
)

# Save the Core ML model
mlmodel.save("model.mlmodel")