import torchvision

class ModelsWithIMAGENET1K_V1:
  '''
  Models with IMAGENET1K_V1 weights
  
  ```
  # Example Usage
  subset_of_models = ModelsWithIMAGENET1K_V1()
  name = 'resnet18'
  model = subset_of_models.init_model(name) # Loads ResNet18
  ```
  '''
  def __init__(self):
    # Get all models + weights
    self.models = {}
    for name in torchvision.models.list_models():
      try:
        weights = self._get_IMAGENET1K_V1_Weights(name)
        self.models[name] = weights
      except ValueError as e:
        print(e)
    # Remove model which doesn't take images of 224 x 224
    self.remove_NonClassificationModels()

  def init_model(self, name: str): # Main functionality
    """
    Args:
    - name (str): Name of model

    Returns:
    - (torch.model): Initalized model obj w/ ImageNet1K_V1 weights
    """
    return torchvision.models.get_model(name, weights='IMAGENET1K_V1')

  def _get_IMAGENET1K_V1_Weights(self, name: str):
    '''
    Get every name of model which has IMAGENET1K_V1 weights
    '''
    all_weights = dir(torchvision.models.get_model_weights(name)) # all weights, lst
    if 'IMAGENET1K_V1' in all_weights:
      return torchvision.models.get_model_weights(name).IMAGENET1K_V1
    else:
      None

  def _remove_non_224x224_Models(self):
    '''
    Delete non 224x224 models from self.models
    '''
    names_of_bad_models = [
        'inception_v3'
    ]
    for name in names_of_bad_models:
      self.models.pop(name, None)

  def remove_NonClassificationModels(self):
    self.remove_SemanticSegmentationModels()
    self.remove_ObjectDetectionModels()
    self.remove_InstanceSegmentationModels()
    self.remove_KeypointDetectionModels()
    self.remove_VideoClassificationModels()
    self.remove_ModelsWithOut224x224()

  def remove_SemanticSegmentationModels(self):
    names = [
        'deeplabv3_mobilenet_v3_large',
        'deeplabv3_resnet101',
        'deeplabv3_resnet50',
        'fcn_resnet50',
        'maskrcnn_resnet50_fpn_v2',
        'fcn_resnet101',
        'lraspp_mobilenet_v3_large'
    ]
    for name in names:
      self.models.pop(name, None)


  def remove_ObjectDetectionModels(self):
    names = [
        'fcos_resnet50_fpn',
        'fasterrcnn_mobilenet_v3_large_320_fpn',
        'fasterrcnn_mobilenet_v3_large_fpn',
        'fasterrcnn_resnet50_fpn',
        'fasterrcnn_resnet50_fpn_v2',
        'retinanet_resnet50_fpn',
        'retinanet_resnet50_fpn_v2',
        'ssd300_vgg16',
        'ssdlite320_mobilenet_v3_large'
    ]
    for name in names:
      self.models.pop(name, None)

  def remove_InstanceSegmentationModels(self):
    names = [
        'maskrcnn_resnet50_fpn_v2',
    ]
    for name in names:
      self.models.pop(name, None)

  def remove_KeypointDetectionModels(self):
    names = [
        'keypointrcnn_resnet50_fpn',
    ]
    for name in names:
      self.models.pop(name, None)

  def remove_VideoClassificationModels(self):
    names = [
        'mc3_18',
        'mvit_v1_b',
        'mvit_v2_s',
        'r2plus1d_18',
        'r3d_18',
        's3d',
        'swin3d_b',
        'swin3d_s',
        'swin3d_t'
    ]
    for name in names:
      self.models.pop(name, None)

  def remove_ModelsWithOut224x224(self):
    for name in subset_of_models.models.keys():
      try:
        crop_size = torchvision.models.get_model_weights(name).IMAGENET1K_V1.transforms.keywords['crop_size']
        if crop_size != 224:
          self.models.pop(name, None)
      except:
        print(f"{name} doesn't have crop_size")
        self.models.pop(name, None)

# Example Usage
subset_of_models = ModelsWithIMAGENET1K_V1()
name = 'resnet18'
model = subset_of_models.init_model(name) # Loads ResNet18