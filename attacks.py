import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import project, project_eps_ball


class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def assert_attack(self, start_x, perturbed_x):
        assert torch.all(start_x + self.eps >= perturbed_x)
        assert torch.all(start_x - self.eps <= perturbed_x)

        assert torch.all(perturbed_x >= 0)
        assert torch.all(perturbed_x <= 1)

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
        performs random initialization and early stopping, depending on the 
        self.rand_init and self.early_stop flags.
        """
        x.requires_grad_()
        attack_x = x
        if self.rand_init:
            rand_init = torch.rand(attack_x.shape) * 2 * self.eps - self.eps # get into [-eps, eps] range
            rand_init = rand_init.to(x.device)
            attack_x = attack_x + rand_init

        # already_stopped = torch.zeros_like(x)
        # attack_res = torch.zeros_like(x)
        ret_attack_x = torch.zeros_like(x)
        done_idx = []

        for i in range(self.n):
            # if results arent good, might need to change this ot work element by element
            grad = torch.autograd.grad(self.loss_func(self.model(x), y).sum(), [x])[0]
            signed_loss_grad = torch.sign(grad)
            if targeted:
                # get x pred closer to y_adv -> make the loss smaller -> gradient descent
                to_project = attack_x - self.alpha * signed_loss_grad
            else:
                # untargeted
                # get x pred farther from y -> make the loss larger -> gradient ascent
                to_project = attack_x + self.alpha * signed_loss_grad

            projected = project(to_project)
            projected = project_eps_ball(projected, x, self.eps)
            attack_x = projected

            # check if attack worked
            attacked_preds = self.model(attack_x)
            attacked_labels = attacked_preds.max(-1)[1]
            if targeted:
                attacked_worked = attacked_labels == y # we want to get adv label
            else:
                attacked_worked = attacked_labels != y # we want to change prediction

            #early stopping
            # we still compute the attack for 'done' x since its all in the same batch
            for k in range(attacked_worked.shape[0]):
                if attacked_worked[k] and torch.all(ret_attack_x[k] == 0):
                    ret_attack_x[k] = copy.deepcopy(attack_x[k].detach())
                    done_idx.append(k)

            if len(done_idx) == attacked_worked.shape[0]:
                self.assert_attack(x, ret_attack_x)
                return ret_attack_x


            # attack_res
        for k in range(ret_attack_x.shape[0]):
            if torch.all(ret_attack_x[k] == 0):
               ret_attack_x[k] = x[k]
        self.assert_attack(x, ret_attack_x)
        return ret_attack_x


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """
    def __init__(self, model, eps=8/255., n=50, alpha=1/255., momentum=0.,
                 k=200, sigma=1/255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma=sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
            and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
            each sample in x.
        """
        pass # FILL ME


class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """
    def __init__(self, models, eps=8/255., n=50, alpha=1/255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss()

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        pass # FILL ME
