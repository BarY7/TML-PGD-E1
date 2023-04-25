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
            rand_init = self.rand_eps(attack_x, x)
            attack_x = attack_x + rand_init

        # already_stopped = torch.zeros_like(x)
        # attack_res = torch.zeros_like(x)
        ret_attack_x = torch.zeros_like(x)
        done_idx = []

        for i in range(self.n):
            # restart
            if i == self.n // 2 and self.rand_init:
                rand_init = self.rand_eps(attack_x, x)
                attack_x = x + rand_init

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


                # equal_idx = projected == attack_x
                # for ind in range(equal_idx.shape[0]):
                #     if torch.all(equal_idx[ind]):
                #         # print(f"reroll index: {ind}")
                #         # print(f"reroll {equal_idx.float().sum()} vals")
                #         random_ball_eps = self.rand_eps(projected[ind], x)
                #         projected[ind] = x[ind] + random_ball_eps



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
                # print("DONE EARLY")
                print(f"Batch SR : {len(done_idx) / x.shape[0]}")
                self.assert_attack(x, ret_attack_x)
                return ret_attack_x


            # attack_res
        for k in range(ret_attack_x.shape[0]):
            if torch.all(ret_attack_x[k] == 0):
               ret_attack_x[k] = attack_x[k]
        self.assert_attack(x, ret_attack_x)
        # print("DONE FUNC###################################")
        print(f"Batch SR : {len(done_idx) / x.shape[0]}")
        return ret_attack_x

    def rand_eps(self, attack_x, x):
        rand_init = torch.rand(attack_x.shape) * 2 * self.eps - self.eps  # get into [-eps, eps] range
        rand_init = rand_init.to(x.device)
        return rand_init


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

    def assert_attack(self, start_x, perturbed_x):
        assert torch.all(start_x + self.eps >= perturbed_x)
        assert torch.all(start_x - self.eps <= perturbed_x)

        assert torch.all(perturbed_x >= 0)
        assert torch.all(perturbed_x <= 1)

    def rand_eps(self, attack_x, x):
        rand_init = torch.rand(attack_x.shape) * 2 * self.eps - self.eps  # get into [-eps, eps] range
        rand_init = rand_init.to(x.device)
        return rand_init

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
        # x.requires_grad_()

        attack_x = x
        if self.rand_init:
            rand_init = torch.rand(attack_x.shape) * 2 * self.eps - self.eps  # get into [-eps, eps] range
            rand_init = rand_init.to(x.device)
            attack_x = attack_x + rand_init

        # already_stopped = torch.zeros_like(x)
        # attack_res = torch.zeros_like(x)
        ret_attack_x = torch.zeros_like(x)
        num_queries = torch.zeros(x.shape[:1])
        done_idx = []

        with_momentum = 0

        for i in range(self.n):
            # if results arent good, might need to change this ot work element by element
            # Changed from whitebox setting
            # approx grad
            # grad = torch.autograd.grad(self.loss_func(self.model(x), y).sum(), [x])[0]
            # change to antithetic sampling and see if results are better

            if i == self.n // 2 and self.rand_init:
                rand_init = self.rand_eps(attack_x, x)
                attack_x = x + rand_init

            # delta = torch.normal(mean=torch.zeros(self.k , *x.shape), std=torch.ones((self.k, *x.shape)))
            delta = torch.randn(self.k)

            #anti
            other_delta_half = - 1 * torch.flip(delta,[0])
            delta = torch.cat([delta,other_delta_half])

            total_sum_batched = 0
            for m in range(delta.shape[0]):
                cur_delta = delta[m].to(x.device)
                reshaped_delta = cur_delta.expand(x.shape[::-1]).reshape(x.shape)
                # reshaped_delta = cur_delta
                theta = x + self.sigma * reshaped_delta.to(x.device)

                area_preds = self.model(theta).detach()
                theta_loss = self.loss_func(area_preds, y).detach()
                inner_sum = reshaped_delta * theta_loss.expand(x.shape[::-1]).reshape(x.shape)
                total_sum_batched += inner_sum

            grad_approx = ( 1 /(self.sigma * self.k * 2) ) * total_sum_batched

            with_momentum = with_momentum * self.momentum + (1 - self.momentum) * grad_approx
            with_momentum = with_momentum.detach()
            del delta


            grad = with_momentum

            # print(torch.cuda.memory_summary())

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

            if torch.any(projected == attack_x):
                equal_idx = projected == attack_x
                for ind in range(equal_idx.shape[0]):
                    if torch.all(equal_idx[ind]):
                        # print(f"reroll index: {ind}")
                        # print(f"reroll {equal_idx.float().sum()} vals")
                        random_ball_eps = self.rand_eps(projected[ind], x)
                        projected[ind] = x[ind] + random_ball_eps

            attack_x = projected

            # check if attack worked
            attacked_preds = self.model(attack_x)
            attacked_labels = attacked_preds.max(-1)[1]
            if targeted:
                attacked_worked = attacked_labels == y  # we want to get adv label
            else:
                attacked_worked = attacked_labels != y  # we want to change prediction

            # early stopping
            # we still compute the attack for 'done' x since its all in the same batch
            for j in range(attacked_worked.shape[0]):
                if attacked_worked[j] and torch.all(ret_attack_x[j] == 0):
                    ret_attack_x[j] = copy.deepcopy(attack_x[j].detach())
                    done_idx.append(j)
                    num_queries[j] = (i + 1) * 2 * self.k

            # print(f"done idx : {len(done_idx)}")

            if len(done_idx) == attacked_worked.shape[0]:
                self.assert_attack(x, ret_attack_x)
                print("DONE EARLY")
                return ret_attack_x, num_queries



            # attack_res
        for j in range(ret_attack_x.shape[0]):
            if torch.all(ret_attack_x[j] == 0):
                ret_attack_x[j] = attack_x[j]
                num_queries[j] = self.n * 2 * self.k

        self.assert_attack(x, ret_attack_x)
        print("DONE")
        print(f"Batch SR : {len(done_idx) / x.shape[0]}")
        return ret_attack_x, num_queries


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
