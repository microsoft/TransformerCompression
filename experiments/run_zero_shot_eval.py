import argparse
import json
import torch
from slicegpt import opt_utils, datautils, utils, opt
from lm_eval import tasks, evaluator, utils
from lm_eval.base import BaseLM
from transformers import OPTForCausalLM, AutoTokenizer
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OPTClass(BaseLM):
    def __init__(self, args, **kwargs):
        # TODO: simplify args and add SliceGPT logic.
        super().__init__()

        class OPTArgs:
            def __init__(self):
                self.model = None
                self.batch_size = 1
                self.eval_dataset = 'wikitext2'
                self.seed = 0
                self.sparsity = None
                self.cal_nsamples = 128
                self.seqlen = None
                self.compress_head = False
                self.load_dir = None


        args = OPTArgs()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = kwargs.get("model")

        self.batch_size_per_gpu = args.batch_size

        self.model = OPTForCausalLM.from_pretrained(self.model_name, torch_dtype='auto').to(self._device)
        self.model.eval()
        self.seqlen = self.model.config.max_position_embeddings
        self.args.seqlen = self.seqlen
        self.args.model = self.model_name
        self.args.sparsity = kwargs.get("sparsity")
        
        # apply slicegpt to model
        self.apply_slicegpt()

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.vocab_size = self.tokenizer.vocab_size
        print('OPT vocab size: ', self.vocab_size)

    def apply_slicegpt(self):
        opt_utils.replace_opt_modules(self.model, self.model.config)
        opt_utils.fuse_opt_modules(self.model)
        print()
        self.model = self.model.cpu()
        
        dataloader, _ = datautils.get_loaders(
            "wikitext2", seed=42, model=self.args.model, seqlen=self.args.seqlen
        )
        
        new_embedding_dimension = int((1 - self.args.sparsity) * self.model.config.hidden_size)
        print(f"New embedding dimension: {new_embedding_dimension} (sparsity {self.args.sparsity})")

        opt.rotate_and_slice_opt(self.model, dataloader, new_embedding_dimension)
        self.model.eval()
        self.model = self.model.to(DEV)

    @classmethod
    def create_from_arg_string(cls, args, kwargs):
        return cls(args, kwargs)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings
    
    @property
    def max_gen_toks(self):
        print('max_gen_toks fn')
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :50272]
        
    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument(
        "--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS)
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--no_cache", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()

    ### SliceGPT ###
    my_model = OPTClass(args, model="facebook/opt-1.3b", sparsity=0.2)

    ### LM Eval Harness ###
    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(
        model=my_model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        no_cache=args.no_cache
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)

    print(evaluator.make_table(results))

if __name__ == "__main__":
    main()