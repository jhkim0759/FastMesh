import torch
import torch.nn.functional as nnf
from torch import nn
from transformers import AutoModelForCausalLM
from models.miche.encode import load_model
from models.shape_opt import ShapeOPTConfig
from models.models import register
from models.vert_refine import VertRefine

def coor_discretize(
    t,
    continuous_range, # (-0.5, 0.5)
    num_discrete: int = 128
):
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo) # 0 <=t < 1
    t *= num_discrete # [0, num_discrete-1]
    assert (t - t.round()).sum() == 0
    assert (t <= num_discrete-1).all() and (t >= 0).all()  # 0 to num_discrete-1

    return t.long()

def undiscretize(
    t,
    low,#-0.5
    high,# 0.5
    num_discrete
):
    assert (t >= 0).all() and (t <= num_discrete-1).all()
    assert high>low
    t = t.float() #[0, num_discrete-1]

    t /= num_discrete  # 0<=t<1
    t = t * (high - low) + low # -0.5 <= t < 0.5
    assert (t < high).all() and (t >= low).all()
    return t

@register("VertGen")
class SingleGPT(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.point_encoder = load_model(ckpt_path=None)
        self.cond_length = 257
        self.cond_dim = 768
        self.block_size = 8 
        self.offset_size = 16

        self.n_discrete_size = self.block_size**3 + self.offset_size**3
        self.pad_id = -1
        self.max_vertices = args['max_vertices']
        self.max_length = int(args['max_vertices'] + 8**3 + self.cond_length + 2) 
        self.gen_max_length = int(args['gen_n_max_triangles'] + self.cond_length + 2) 

        vocab_size = self.n_discrete_size + 3 # 4 for bos, eos, pad, &
        self.config = ShapeOPTConfig.from_pretrained(
            args['llm'],
            n_positions=self.max_length,
            max_position_embeddings=self.max_length,
            vocab_size = vocab_size,
            _attn_implementation="flash_attention_2"
        )

        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

        self.config.bos_token_id = self.bos_token_id
        self.config.eos_token_id = self.eos_token_id
        self.config.pad_token_id = self.pad_token_id
        self.config._attn_implementation ="flash_attention_2"
        self.config.n_discrete_size = self.n_discrete_size
        self.config.cond_length = self.cond_length
        self.config.max_vertices = args['max_vertices']
        self.config.word_embed_proj_dim = self.config.hidden_size

        self.transformer = AutoModelForCausalLM.from_config(
            config=self.config, use_flash_attention_2 = True
        )

        self.cond_head_proj = nn.Linear(self.cond_dim, self.config.word_embed_proj_dim)
        self.cond_proj = nn.Linear(self.cond_dim * 2, self.config.word_embed_proj_dim)
        
        self.train()

        self.use_refine = args['use_refine']
        if args['use_refine']:
            self.refine_net = VertRefine()
            self.refine_net.eval()
            for param in self.refine_net.parameters():
                param.requires_grad = False


    def detokenize(self, input_ids):
        input_ids = input_ids.reshape(input_ids.shape[0], -1) # B x L
        batch_size = input_ids.shape[0]

        vert_list = []
        for i in range(batch_size):
            sequence = input_ids[i]
            sequence = sequence[sequence != self.pad_id]

            block_coord = torch.tensor([0,0,0])
            detokenized = []
            for token in sequence:
                if token<0:
                    break
                if token< self.block_size**3:
                    x = token//self.block_size**2
                    y = (token%self.block_size**2)//self.block_size
                    z = token%self.block_size
                    block_coord = torch.tensor([x,y,z])*self.offset_size
                    continue
                token -= self.block_size**3
                x = token//self.offset_size**2
                y = (token%self.offset_size**2)//self.offset_size
                z = token%self.offset_size
                
                detokenized.append(torch.tensor([x,y,z])+block_coord)
            
            detokenized = torch.stack(detokenized) if detokenized !=[] else torch.empty((0,3))
            
            vert_list.append(detokenized / (self.block_size*self.offset_size) - 0.5) 

        return vert_list 

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self,"point_encoder"):
            self.point_encoder.eval()
            for param in self.point_encoder.parameters():
                param.requires_grad = False

    def forward(self, data_dict: dict, is_eval: bool = False) -> dict:
        if not is_eval:
            return self.train_one_step(data_dict)
        else:
            return self.generate(data_dict)

    def pad_id_and_attn(self, input_ids, attention_mask, face_ids = None): # same
        # reserve one space for `bos`, the pad_id will be replaced to `bos`
        place_holder = torch.ones_like(input_ids[:, [0]])   # batch x 1
        # prepare input_ids and attention_mask for transformers
        input_ids[attention_mask.bool()] += 3 # 0 - num_tokens to 3 - num_tokens + 3, total: 0 - num_tokens + 3, num: numtokens + 4
        input_ids[~attention_mask.bool()] = self.pad_token_id # in transformers pad token id is only used for init nn.embedding which we won't use
        
        input_ids = torch.cat(
            (place_holder * self.bos_token_id, input_ids, place_holder * self.pad_token_id),
            dim=1
        )
        input_ids[torch.arange(0, input_ids.shape[0]), attention_mask.sum(dim=1).long()+1] = self.eos_token_id

        attention_mask = torch.cat(
            (place_holder, place_holder, attention_mask, ),
            dim=1
        )
        # length
        return input_ids, attention_mask

    def process_point_feature(self, point_feature):
        encode_feature = torch.zeros(point_feature.shape[0], self.cond_length, self.config.word_embed_proj_dim,
                                    device=self.cond_head_proj.weight.device, dtype=self.cond_head_proj.weight.dtype)
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(torch.cat([point_feature[:, 1:], shape_latents], dim=-1))

        return encode_feature

    def train_one_step(self, data_dict: dict):
        point_feature = self.point_encoder.encode_latents(data_dict["pc_normal"])
        with torch.no_grad():
            assert "sequence" in data_dict
            input_ids = data_dict['vert_sequence']
            attention_mask = input_ids != self.pad_id

            sequence_max_length = attention_mask.sum(dim=1).max()
            input_ids = input_ids[:, :sequence_max_length]
            attention_mask = attention_mask[:, :sequence_max_length]

            input_ids, attention_mask = self.pad_id_and_attn(input_ids, attention_mask, face_ids=None)

        # add cond_length to attention mask
        pad_attention_mask = torch.ones((attention_mask.shape[0], self.cond_length), device=attention_mask.device, dtype=attention_mask.dtype)
        attention_mask = torch.concatenate((pad_attention_mask, attention_mask), dim=1)

        processed_point_feature = self.process_point_feature(point_feature=point_feature)
        
        output = self.transformer(
            inputs_embeds = processed_point_feature,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True
        )
        
        # compute loss with shift token right
        logit = output.logits[:, self.cond_length-1:-1]  # batch x ntoken x vocab
        label = input_ids[:, 0:]  # batch x ntoken
        masks = attention_mask[:, self.cond_length-1:-1]  # batch x ntoken
        # also predict bos token
        loss_per_token = nnf.cross_entropy(
            logit.permute(0, 2, 1),  # batch x vocab x ntoken
            label,
            reduction='none'
        )  # batch x ntoken
        
        final_loss = torch.sum(loss_per_token * masks) / (torch.sum(masks) + 1e-8)
        data_dict['loss'] = final_loss

        return data_dict


    @torch.no_grad()
    def generate(self, data_dict) -> dict:
        point_feature = self.point_encoder.encode_latents(data_dict["pc_normal"])
        processed_point_feature = self.process_point_feature(point_feature)
        generate_length = self.gen_max_length - self.cond_length
        net_device = next(self.parameters()).device
        outputs = torch.ones(point_feature.shape[0], generate_length).long().to(net_device) * self.eos_token_id
        # batch x ntokens
        
        results = self.transformer.generate(
            inputs_embeds = processed_point_feature,
            max_new_tokens = generate_length, # all faces plus two
            do_sample=True,
            temperature = self.args['temp'],
            top_k=self.args['top_k'],
            top_p=self.args['top_p'],
            bos_token_id = self.bos_token_id,
            eos_token_id = self.eos_token_id,
            pad_token_id = self.pad_token_id,
            return_dict_in_generate = True,
        )

        assert results.sequences.shape[1] <= generate_length # B x ID  bos is not included since it's predicted

        outputs[:, :results.sequences.shape[1]] = results.sequences
        # batch x ntokens ====> batch x ntokens x D
        outputs = outputs[:, 1: -1] # eos and bos removed

        outputs[outputs == self.bos_token_id] = self.pad_id
        outputs[outputs == self.eos_token_id] = self.pad_id
        outputs[outputs == self.pad_token_id] = self.pad_id
        outputs[outputs != self.pad_id] -= 3
        
        gen_vert = self.detokenize(outputs)

        if self.use_refine:
            refined_vertices = self.refine_net.refine_vertices(gen_vert, point_feature)
            gen_vert = torch.nn.utils.rnn.pad_sequence(gen_vert, batch_first=True, padding_value=-1, padding_side='right').to(point_feature.device)
        else:
            refined_vertices = torch.nn.utils.rnn.pad_sequence(gen_vert, batch_first=True, padding_value=-1, padding_side='right').to(point_feature.device)

        return refined_vertices, gen_vert