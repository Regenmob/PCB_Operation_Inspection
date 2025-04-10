import openvino as ov

class ModelManager:
    def __init__(self, model_path, device="CPU"):
        self.core = ov.Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        self.input_layer = self.compiled_model.input(0)
        self.output_boxes = self.compiled_model.output("boxes")
        self.output_labels = self.compiled_model.output("labels")

    def infer(self, input_tensor):
        result = self.compiled_model([input_tensor])
        return result[self.output_boxes], result[self.output_labels]
