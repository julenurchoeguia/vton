### Local imports ###

import json
from scm.src.scm import SCM
from scm.src.scm_from_article import NAFNet_Combine
import torch

scm_model = SCM(img_channel=6, width=8, middle_blk_num=2, enc_blk_nums=[2, 4], dec_blk_nums=[2, 2])
scm_reference = NAFNet_Combine(img_channel=6, width=8, middle_blk_num=2, enc_blk_nums=[2, 4], dec_blk_nums=[2, 2])

with open('scm/scripts/mapping_dict.json') as f:
    mapping_dict = json.load(f)

state_dict = scm_model.state_dict()
state_dict_reference = scm_reference.state_dict()

for key in state_dict_reference.keys():
    new_key = mapping_dict[key]
    if "Parameter" in new_key:
        state_dict[new_key] = state_dict_reference[key].squeeze(0)
    else:
        state_dict[new_key] = state_dict_reference[key]

scm_model.load_state_dict(state_dict)


random_input = torch.rand(1, 6, 128, 128)

output = scm_model(random_input)
output_reference = scm_reference(random_input)

validation = torch.allclose(output, output_reference, atol=1e-5)

if validation:
    print("Validation passed")
else:
    print("Validation failed")
