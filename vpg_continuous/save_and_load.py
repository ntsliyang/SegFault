import os
import torch
from torch import optim
from model import PolicyNet, ValueNet


# Utils for saving and loading checkpoints

def save_checkpoint(file_dir, policy_net, value_net, policynet_optim, valuenet_optim,
                    i_epoch, policy_lr, valuenet_lr, **kwargs):
    save_dict = {"policy_net": policy_net.state_dict(),
                 "value_net": value_net.state_dict(),
                 "policynet_optim": policynet_optim.state_dict(),
                 "valuenet_optim": valuenet_optim.state_dict(),
                 "i_epoch": i_epoch,
                 "policy_lr": policy_lr,
                 "valuenet_lr": valuenet_lr
                 }
    # Save optional contents
    save_dict.update(kwargs)

    # Create the directory if not exist
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

    file_name = os.path.join(file_dir, "ckpt_eps%d.pt" % i_epoch)

    # Delete the file if the file already exist
    try:
        os.remove(file_name)
    except OSError:
        pass

    # Save the file
    torch.save(save_dict, file_name)


def load_checkpoint(file_dir, i_epoch, layer_sizes, input_size, action_lim, device='cuda'):
    checkpoint = torch.load(os.path.join(file_dir, "ckpt_eps%d.pt" % i_epoch), map_location=device)

    policy_net = PolicyNet(layer_sizes, action_lim).to(device)
    value_net = ValueNet(input_size).to(device)
    policy_net.load_state_dict(checkpoint["policy_net"])
    policy_net.train()
    value_net.load_state_dict(checkpoint["value_net"])
    value_net.train()

    policy_lr = checkpoint["policy_lr"]
    valuenet_lr = checkpoint["valuenet_lr"]

    policynet_optim = optim.Adam(policy_net.parameters(), lr=policy_lr)
    policynet_optim.load_state_dict(checkpoint["policynet_optim"])
    valuenet_optim = optim.Adam(value_net.parameters(), lr=valuenet_lr)
    valuenet_optim.load_state_dict(checkpoint["valuenet_optim"])

    checkpoint.pop("policy_net")
    checkpoint.pop("value_net")
    checkpoint.pop("policynet_optim")
    checkpoint.pop("valuenet_optim")
    checkpoint.pop("i_epoch")
    checkpoint.pop("policy_lr")
    checkpoint.pop("valuenet_lr")

    return policy_net, value_net, policynet_optim, valuenet_optim, checkpoint
