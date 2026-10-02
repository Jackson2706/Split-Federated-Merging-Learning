from .CNN_ISIC import ISIC_UNET_2D

model_dataset_map = {"resnet50": {"isic-2018": ISIC_UNET_2D}}


def get_model(model, dataset):
    try:
        return model_dataset_map[model][dataset]
    except KeyError:
        exit("Error: unrecognized model")
