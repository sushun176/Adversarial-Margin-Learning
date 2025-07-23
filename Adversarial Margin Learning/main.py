import time
from transformers import BertForSequenceClassification
import torch
from tqdm import tqdm
from model import Transformer
from config import get_config
import torch.nn.functional as F
from data_utils import load_data
from transformers import logging, AutoTokenizer, AutoModel
from ad import FGM, PGD
from torch.optim import lr_scheduler
import os
import torch
from loss_func import AMLLoss
class Instructor:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            base_model = AutoModel.from_pretrained('bert-base-uncased',output_attentions=True)
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            base_model = AutoModel.from_pretrained('roberta-base')

        elif args.model_name == 'ernie':
            self.tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-2.0-base-en', add_prefix_space=True)
            base_model = AutoModel.from_pretrained('nghuyong/ernie-2.0-base-en')
        else:
            raise ValueError('unknown model')
        self.model = Transformer(base_model, args.num_classes, args.method)
        self.model.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()
    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

   
    def _train(self, dataloader, criterion, optimizer):
        fgm = FGM(self.model, epsilon=1.0, emb_name='embeddings.word_embeddings.weight')
        train_loss, n_correct, n_train = 0, 0, 0
        self.model.train()
        for inputs, targets,aa in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = (1-self.args.lamda)*F.cross_entropy(outputs["predicts"], targets)+self.args.lamda*criterion(outputs, targets)
            loss.backward()
            fgm.attack()  
            predictions_adv = self.model(inputs)
            loss_adv = (1-self.args.lamda)*F.cross_entropy(predictions_adv["predicts"], targets)+self.args.lamda*criterion(predictions_adv, targets)
            loss_adv.backward() 
            fgm.restore()
            optimizer.step()
            self.model.zero_grad()

            train_loss = train_loss + loss.item() * targets.size(0)
            n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
            n_train += targets.size(0)

        return train_loss / n_train, n_correct / n_train


    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            for inputs, targets ,tokens in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = (1-self.args.lamda)*F.cross_entropy(outputs["predicts"], targets)+self.args.lamda*criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
                n_test += targets.size(0)


        return test_loss / n_test, n_correct / n_test

    def run(self):
        train_dataloader, test_dataloader = load_data(dataset=self.args.dataset,
                                                      data_dir=self.args.data_dir,
                                                      tokenizer=self.tokenizer,
                                                      train_batch_size=self.args.train_batch_size,
                                                      test_batch_size=self.args.test_batch_size,
                                                      model_name=self.args.model_name,
                                                      method=self.args.method,
                                                      workers=0)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())

        criterion = AMLLoss()

        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay)
        scheduler = lr_scheduler.CosineAnnealingLR(
           optimizer, T_max=self.args.num_epoch, eta_min=self.args.min_lr)
        best_loss, best_acc = 0, 0
        best_model_path = 'best_model.pth'
        for epoch in range(self.args.num_epoch):
            if len(train_dataloader) == 0:
                raise ValueError("The training dataloader is empty. Please check your dataset and dataloader.")
            train_loss, train_acc= self._train(train_dataloader ,criterion, optimizer)
            test_loss, test_acc = self._test(test_dataloader,criterion)
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.info(f'Model weights saved to {best_model_path}')
            self.logger.info('{}/{} - {:.2f}%'.format(epoch+1, self.args.num_epoch, 100*(epoch+1)/self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc*100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc*100))
            self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc*100))
            self.logger.info('current lr:{:.6f}'.format(current_lr))
        self.logger.info('log saved: {}'.format(self.args.log_name))


if __name__ == '__main__':


    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    logging.set_verbosity_error()
    args, logger = get_config()
    ins = Instructor(args, logger)
    ins.run()
