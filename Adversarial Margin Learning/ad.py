

import torch
import torch

class AdaptiveFGM:
    def __init__(self, model,emb_name, epsilon=0.1, gradient_threshold=0.5):
        self.model = model
        self.base_epsilon = epsilon
        self.gradient_threshold = gradient_threshold
        self.backup = {}
        self.emb_name = emb_name

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                grad = torch.autograd.grad(outputs=self.model, inputs=param,
                                           grad_outputs=torch.ones_like(self.model.output),
                                           create_graph=True)[0]
                grad_norm = torch.norm(grad)
                adaptive_epsilon = self.base_epsilon * (grad_norm / self.gradient_threshold)
                if grad_norm != 0:
                    r_at = adaptive_epsilon * grad / grad_norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class FGM():
    '''
    Example
    # 初始化
    fgm = FGM(model,epsilon=1,emb_name='word_embeddings.')
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack() # 在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
    '''
    def __init__(self, model,emb_name,epsilon=0.9):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm!=0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

                #print("epsilon:",self.epsilon)
                # print(f"Applied FGM attack on {name}")
                # print(f"Parameter values after attack: {param.data[:5]}")  # 打印前5个值

                # if param.requires_grad:
                #     print(f' Parameter {name}, Gradient Norm: {param.grad.norm().item()}')
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
                # print(f"Restored original parameters for {name}")
                # print(f"Parameter values after restore: {param.data[:5]}")  # 打印前5个值
        self.backup = {}

class PGD():
    '''
    Example
    pgd = PGD(model,emb_name='word_embeddings.',epsilon=1.0,alpha=0.3)
    K = 3
    for batch_input, batch_label in data:

        loss = model(batch_input, batch_label)
        loss.backward()
        pgd.backup_grad()

        for t in range(K):
            pgd.attack(is_first_attack=(t==0))
            if t != K-1:
                model.zero_grad()
            else:
                pgd.restore_grad()
            loss_adv = model(batch_input, batch_label)
            loss_adv.backward()
        pgd.restore()

        optimizer.step()
        model.zero_grad()
    '''
    def __init__(self, model,emb_name,epsilon=1.,alpha=1.0):

        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self,is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()

                if param.grad is None:
                  #  print("Skipping parameter with no gradient.")
                    continue
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.grad is None:

                continue
            else:
                self.grad_backup[name] = param.grad.clone()
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:

                continue
            else:
                param.grad = self.grad_backup[name]
class PGD:
    def __init__(self, model, emb_name, epsilon=1.0, alpha=0.3):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.epsilon:
            r = self.epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
