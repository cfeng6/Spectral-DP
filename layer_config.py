import torch
import os
import privacy_budget
from configs import set_arguments


args = set_arguments()
if torch.cuda.is_available():
     args.gpu = True


DEVICE = torch.device('cuda') if args.gpu else torch.device('cpu')
TOTAL_BOUND = args.max_grad_norm
LAYER_BOUND = TOTAL_BOUND
FILTER_RATIO = args.filter_ratio

sub_root = os.path.join(args.root)

SIGMA = privacy_budget.calculate_sigma(
    args.dataset, 
    sub_root, 
    DEVICE, 
    args.epochs, 
    args.epsilon, 
    args.train_batch_size, 
    args.delta, 
    args.max_grad_norm)

if args.model == 'mlp_mnist':
    num_layers = 8
    LAYER_BOUND = (TOTAL_BOUND ** 2 / num_layers)**0.5
else:
    raise Exception("Not a training model")






