from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential


def get_ml_client(
    subscription_id="7f583e60-37b4-432d-9990-ad8df0bff2ca",
    resource_group="aml-neural-test",
    workspace_name="aml-neural",
    credential=DefaultAzureCredential(),
):

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    return ml_client


class VCInfo:
    def __init__(
        self, subscription_id="156138e5-a3b1-48af-9e2e-883f4df3f457", resource_group="gcr-singularity-lab", vc="dell1"
    ):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.vc = vc
        self.compute_config = (
            "/subscriptions/"
            + subscription_id
            + "/resourceGroups/"
            + resource_group
            + "/providers/Microsoft.MachineLearningServices/virtualclusters/"
            + vc
        )


def get_kwargs(vc="baltic05", num_gpus=1, image="acpt-pytorch-2.0.1-py3.10-cuda11.8"):
    if vc in [
        "dell1",
        "kings01",
        "kings02",
        "kings03",
        "kings04",
        "kings05",
        "kings06",
        "kings07",
        "kings08",
        "kings09",
        "kings10",
        "kings11",
        "kings12",
        "mckinley01",
        "mckinley02",
        "mckinley03",
        "mckinley04",
        "mckinley05",
        "mckinley06",
        "mckinley07",
        "mckinley08",
        "barlow01",
        "barlow02",
        "barlow03",
        "barlow04",
        "barlow05",
        "barlow06",
        "barlow07",
        "barlow08",
        "barlow09",
        "msrresrchlab",
    ]:
        vc_info = VCInfo(
            subscription_id="156138e5-a3b1-48af-9e2e-883f4df3f457", resource_group="gcr-singularity-lab", vc=vc
        )
    elif vc in [
        "baltic01",
        "baltic02",
        "baltic03",
        "baltic04",
        "baltic05",
        "baltic06",
        "baltic07",
        "baltic08",
        "baltic09",
        "baltic10",
        "baltic11",
        "baltic12",
        "huashanvc1",
        "huashanvc2",
        "huashanvc3",
        "huashanvc4",
    ]:
        vc_info = VCInfo(
            subscription_id=" 22da88f6-1210-4de2-a5a3-da4c7c2a1213", resource_group="gcr-singularity", vc=vc
        )
    elif vc in ["msrresrchvc"]:
        vc_info = VCInfo(
            subscription_id=" 22da88f6-1210-4de2-a5a3-da4c7c2a1213", resource_group="gcr-singularity-resrch", vc=vc
        )
    elif vc in ["msroctovc"]:
        vc_info = VCInfo(
            subscription_id=" d4404794-ab5b-48de-b7c7-ec1fefb0a04e", resource_group="gcr-singularity-octo", vc=vc
        )

    if vc == 'baltic05':
        if num_gpus == 1:
            instance_type = "Singularity.ND12_H100_v5"
        elif num_gpus == 2:
            instance_type = "Singularity.ND24_H100_v5"
        elif num_gpus == 4:
            instance_type = "Singularity.ND48_H100_v5"
        elif num_gpus == 8:
            instance_type = "Singularity.ND96r_H100_v5"
        else:
            raise ValueError(f"baltic05 does not support {num_gpus} GPUs")
    elif vc == 'huashanvc2':
        if num_gpus == 2:
            instance_type = "Singularity.ND12as_MI200_v4"
        elif num_gpus == 4:
            instance_type = "Singularity.ND24as_MI200_v4"
        elif num_gpus == 8:
            instance_type = "Singularity.ND48as_MI200_v4"
        elif num_gpus == 16:
            instance_type = "Singularity.ND96as_MI200_v4"
        else:
            raise ValueError(f"huashanvc2 does not support {num_gpus} GPUs")
    else:
        raise NotImplementedError

    resources = {
        "instance_type": instance_type,
        "instance_count": 1,
        "properties": {
            "AISuperComputer": {
                "interactive": False,
                "imageVersion": image,
                "slaTier": "Premium",
                "scalePolicy": {
                    "autoScaleIntervalInSec": 120,
                    "maxInstanceTypeCount": 1,
                    "minInstanceTypeCount": 1,
                },
            }
        },
    }
    environment = Environment(
        image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210513.v1",
        # This is a shadown image which is workaround
        # This image will not be used.
    )

    return {"resources": resources, "compute": vc_info.compute_config, "environment": environment}
