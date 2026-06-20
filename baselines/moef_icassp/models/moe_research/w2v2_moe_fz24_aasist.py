import sys
sys.path.append("./")
import os
from transformers import Wav2Vec2Model,AutoConfig
import torch.nn as nn
import torch.nn.functional as F
import torch
from models.wav2vec.aasist import W2VAASIST
from models.moe_research.MoLE import MoELocal, MoE24fusion
    

class this_arg():
    stage = 1
    moe_topk = 2
    # experts per feature
    moe_experts = 4
    moe_exp_hid = 128
    # if stage == 1, means that the pretrained model is freezed
    # if stage == 2, means that training will finetune the pretrained model

class Model(nn.Module):
    def __init__(self, args = this_arg()):
        super().__init__()
        pretrained_path = (
            os.environ.get("MOEF_WAV2VEC2_PATH")
            or os.environ.get("MOEF_XLSR_HF_PATH")
            or "/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr_300m"
        )
        print(f"Using wav2vec2 checkpoint: {pretrained_path}")
        self.pretrain_model = Wav2Vec2Model.from_pretrained(
            pretrained_path)
        for param in self.pretrain_model.parameters():
            param.requires_grad = False
        self.classifier = W2VAASIST()
        self.moe_l = MoE24fusion(
            ds_inputsize=1024,
            input_size=1024,
            output_size=1024,
            num_experts=24*args.moe_experts,
            hidden_size=args.moe_exp_hid, 
            noisy_gating=True,
            k = args.moe_topk,
            trainingmode=True
            )
    
    def forward(self, x,train = False):
        with torch.no_grad():
        # if True:
            x = self.pretrain_model(
                x,
                output_hidden_states = True,
                output_attentions = True
                )
        bs,t,sp = x[0].shape
        hidden_ones = []
        for i in range(24):
            hidden_ones.append(x.hidden_states[i].view(bs*t, sp))
        
        fusion_x = self.moe_l(
            x.last_hidden_state.view(bs*t, sp),
            hidden_ones,
            training = train
            )
        
        pred , hidden_state = self.classifier(fusion_x[0].view(bs,t, sp))
        return pred , (x.hidden_states,x.attentions), fusion_x[1]
        
        
if __name__ == "__main__":
    md = Model()
    # md.freeze_parameters()
    # md.unfreeze_parameters()
    # op, hd,_ = md( torch.randn((2,64600)))
    # print(op.shape)
    print(sum(p.numel() for p in md.pretrain_model.parameters() if p.requires_grad)/1000000)
