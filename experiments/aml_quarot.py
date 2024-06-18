import os
from pathlib import Path

from aml_util import get_kwargs, get_ml_client
from azure.ai.ml import command
from azure.ai.ml.sweep import Choice, LogUniform, Uniform

kwargs = get_kwargs(vc="baltic05", num_gpus=2, image="acpt-2.2.1-py3.10-cuda12.1")
#kwargs = get_kwargs(vc="huashanvc2", num_gpus=2, image="rocm-py-pytorch-deepspeed")
ml_client = get_ml_client()

job = command(
    inputs=dict(
        gptq_damping=0.01,
        cal_nsamples=128,
        groupsize=128,
        rotate=True,
        asym=False,
        model="mistralai/Mixtral-8x7B-v0.1",
        clip_ratio=0.9,
        bits=8,
    ),
    code=Path(__file__).parent.parent,
    command=(
        "pip install -e .[finetune];"
        "pip install -e .[quarot];"
        "pip install packaging;"
        "pip install flash-attn --no-build-isolation;"
        "wandb login --relogin --host=https://microsoft-research.wandb.io;"
        "python experiments/run_quarot.py"
        " --rotate ${{inputs.rotate}}"
        # weight stuff
        " --w-gptq"
        " --gptq-damping ${{inputs.gptq_damping}}"
        " --w-bits ${{inputs.bits}}"
        " --w-groupsize ${{inputs.groupsize}}"
        " --w-asym ${{inputs.asym}}"
        # " --gptq-opt-scales" very slow, makes little difference!
        #activation stuff:
        " --a-bits ${{inputs.bits}}"
        " --a-clip-ratio ${{inputs.clip_ratio}}"
        " --a-groupsize ${{inputs.groupsize}}"
        " --a-asym ${{inputs.asym}}"
        # #kv stuff
        # " --v-bits ${{inputs.bits}}"
        # " --v-clip-ratio 1.0"
        # " --k-bits ${{inputs.bits}}"
        # " --k-clip-ratio 1.0"
        # model stuff
        " --model ${{inputs.model}}"
        " --cal-nsamples ${{inputs.cal_nsamples}}"
        " --cal-batch-size 16"
        " --lm-eval"
        " --wandb-project quarot-phi-wna"
        " --distribute-model"
    ),
    environment_variables={
        'JOB_EXECUTION_MODE': "basic",
        'AZUREML_COMPUTE_USE_COMMON_RUNTIME': 'true',
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
        "HF_TOKEN": os.getenv("HF_TOKEN"),
    },
    display_name="quarot-wna-sweep-phi",
    tags=dict(ProjectID="PRJ-0001-A04", Experiment="quarot-mixtral"),
    **kwargs
)


sweep_job = job(
    rotate=Choice([1, 0]),
    asym=Choice([0, 1]),
    groupsize=Choice([0, 32, 64]),
    clip_ratio=Choice([0.7, 0.75, 0.8, 0.85, 0.9, 0.95]),
    cal_nsamples=Choice([128]),
    model=Choice(
            [
            #"mistralai/Mixtral-8x7B-v0.1"]
            "microsoft/Phi-3-mini-4k-instruct",
            #"meta-llama/Meta-Llama-3-8B"
            ]     
        ),
    # model = "meta-llama/Meta-Llama-3-8B","meta-llama/Llama-2-7b-hf"
    bits=Choice([2,3, 4]),
).sweep(compute=kwargs['compute'], sampling_algorithm='grid', primary_metric='quarot_ppl', goal="Minimize")
sweep_job.resources = kwargs['resources']
returned_job = ml_client.jobs.create_or_update(sweep_job)
print(returned_job.studio_url)
