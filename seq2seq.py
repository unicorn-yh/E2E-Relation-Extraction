import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import pandas as pd
import re
import torchtext


# 常量
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")  # 正则表达式模式
TRAIN_FNAME = "e2e_dataset/trainset.csv" 
DEV_FNAME = "e2e_dataset/devset.csv"
TEST_FNAME = "e2e_dataset/testset.csv"
DATA_LEN_HIST_PATH = "result/data_len_hist.png"
SRC_VOCAB = []
TGT_VOCAB = []
MAX_LEN = 50
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'


'''Config类用于设置实验参数'''
class Config:
    def __init__(self):
        self.save_data_path = "result/result.txt"
        self.encoder_save_path = "model/encoder.mdl"
        self.decoder_save_path = "model/decoder.mdl"
        self.plot_save_path = "result/loss_plot.png"
        self.input_dim = len(SRC_VOCAB)
        self.output_dim = len(TGT_VOCAB)
        self.hidden_size = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = 0.01
        self.max_len = MAX_LEN
        self.batch_size = 1    
        self.max_epoch = 1


'''数据分析'''
class Data_analysis():
    def __init__(self, save_path=DATA_LEN_HIST_PATH):
        data = pd.read_csv(TRAIN_FNAME,dtype=str) 
        references = [self.process_e2e_text(data.iloc[i]['ref']) for i in range(len(data))]
        self.references_lens = [len(d) for d in references]
        self.plot_len_hist(self.references_lens, save_path)

    def process_e2e_text(self,s):   # 将每个ref表示为token列表。
        words = []
        for fragment in s.strip().split():
            fragment_tokens = _WORD_SPLIT.split(fragment)
            words.extend(fragment_tokens)
        tokens = [w for w in words if w]
        return tokens

    def cnt_bins_and_cnts(self):   # 用于计算句子长度的确切数量
        lengths_to_consider = [0,10,20,30,40,50,60,70,80]
        bins = [(lengths_to_consider[i], lengths_to_consider[i+1]) \
                for i in range(len(lengths_to_consider)-1)]
        cnts = [0] * len(bins)
        for l in self.references_lens:
            for bin_idx, b in enumerate(bins):
                if l > b[0] and l <= b[1]:
                    cnts[bin_idx] += 1
                    break
        return (bins, cnts)

    def plot_len_hist(self, lens, fname):   # 绘制参考长度分布直方图
        references_lens_df = pd.DataFrame(self.references_lens)
        mean = float(references_lens_df.mean())   # 检索统计数据
        std = float (references_lens_df.std())
        min_len = int(references_lens_df.min())
        max_len = int(references_lens_df.max())

        # 绘制长度分布的直方图
        plt.figure(0)    
        plt.rcParams.update({'font.size': 12})
        n, bins, patches = plt.hist(lens, 20, facecolor='b', alpha=0.55)
        plt.xlabel('Sentence Length')
        plt.ylabel('Number of sentences')
        plt.title('Sentence length distribution')
        plt.axis([0, 80, 0, 10000])
        plt.text(40, 7500, r'$mean={:.2f},\ std={:.2f}$'.format(mean, std))
        plt.text(40, 6800, r'$min={},\ max={}$'.format(min_len, max_len))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()


'''读取数据'''
class E2EDataset(Dataset):
    def __init__(self):
        '''
            __init__()函数的内容：
            1.读取数据集
            2.对读取的数据表做序列化处理
            3.构建输入(序列化后的数据表)和输出(参考文本)的词典
        '''
        src = torchtext.data.Field(
            batch_first=True, 
            include_lengths=True
            )
        tgt = torchtext.data.Field(
            batch_first=True, 
            preprocessing = lambda seq: [SOS_TOKEN] + seq + [EOS_TOKEN]
            )
        self.train_data = torchtext.data.TabularDataset(
                path=TRAIN_FNAME, format='csv',
                fields=[('src', src), ('tgt', tgt)],
                filter_pred=self.len_filter
            )
        self.dev_data = torchtext.data.TabularDataset(
                path=DEV_FNAME, format='csv',
                fields=[('src', src), ('tgt', tgt)],
                filter_pred=self.len_filter
            )
        self.test_data = torchtext.data.TabularDataset(
                path=TEST_FNAME, format='csv',
                fields=[('src', src)],
                filter_pred=self.len_filter_test
            )
        src.build_vocab(self.train_data.src, self.dev_data.src, self.test_data.src, 
                        max_size=50000)
        tgt.build_vocab(self.train_data.tgt, self.dev_data.tgt, self.test_data.tgt, 
                        max_size=50000)
        self.src_vocab = src.vocab
        self.tgt_vocab = tgt.vocab
        self.print_dataset_info(self.train_data, src, tgt, self.src_vocab, self.tgt_vocab)
        
    def len_filter(self, example):
        return len(example.src) <= MAX_LEN and len(example.tgt) <= MAX_LEN

    def len_filter_test(self, example):
        return len(example.src) <= MAX_LEN
    
    def print_dataset_info(self, data, src, tgt, input_vocab, output_vocab):
        print('20 tokens from input vocab:\n', list(input_vocab.stoi.keys())[:20])
        print('\n20 tokens from output vocab:\n', list(output_vocab.stoi.keys())[:20])
        print('\nnum training examples:', len(data.examples))
        item = random.choice(data.examples)
        print('\nexample train data:')
        print('src:\n', item.src)
        print('tgt:\n', item.tgt)
        print("\nSRC VOCAB LEN:",len(input_vocab))
        print("TRG VOCAB LEN:",len(output_vocab))
        print()


'''编码器'''
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        '''
            在__init__()函数初始化使用的网络
        '''
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, myinput, hidden):
        '''
            1.forward()函数接收输入的id序列，送入Encoder，编码返回最后
        一个token对应的隐藏状态用于初始化Decoder的隐藏状态
            2.如果要采用Attention，Encoder需返回所有token对应的隐藏状态
        '''
        embedded = self.embedding(myinput)
        embedded = embedded.view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=config.device)


'''解码器'''
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        '''
            在__init__()函数初始化使用的网络
        '''
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input, hidden):
        '''
            1.forward()函数以<sos>对应的id作为第一个输入，t时刻的输入为
        t-1时刻的输出，解码出预测序列第t个token
            2.解码过程迭代至预测出<eos>这一token，或者达到预设最大长度结束
            3.如果采用Attention，需要在Decoder计算Attention Score
        '''
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=config.device)
    

'''Seq2seq模型训练'''
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LEN, teacher_forcing_ratio=0.5):
    
    # get an initial hidden state for the encoder
    encoder_hidden = encoder.initHidden()

    # zero the gradients of the optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # get the seq lengths, used for iterating through encoder/decoder
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # create empty tensor to fill with encoder outputs
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=config.device)

    # create a variable for loss
    loss = 0
    
    # pass the inputs through the encoder
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # create a start-of-sequence tensor for the decoder
    decoder_input = torch.tensor([[TGT_VOCAB.stoi[SOS_TOKEN]]], device=config.device)

    # set the decoder hidden state to the final encoder hidden state
    decoder_hidden = encoder_hidden

    # decide if we will use teacher forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if use_teacher_forcing:
            decoder_input = target_tensor[di]
        if decoder_input.item() == TGT_VOCAB.stoi[EOS_TOKEN]:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


'''执行迭代训练'''
def trainIters(data_train, encoder, decoder, n_iters, print_every=1000, learning_rate=0.01, teacher_forcing_ratio=0.5):
    print(f'Running {n_iters} epochs...')
    print_loss_total = 0
    print_loss_epoch = 0
    loss_ls = []

    encoder_optim = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optim = optim.SGD(decoder.parameters(), lr=learning_rate)

    batch_iterator = torchtext.data.Iterator(
        dataset=data_train, batch_size=config.batch_size,
        sort=False, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=config.device, repeat=False)
    
    criterion = nn.NLLLoss()

    for e in range(n_iters):
        batch_generator = batch_iterator.__iter__()
        step = 0
        start = time.time()

        for batch in batch_generator:
            step += 1
            
            # get the input and target from the batch iterator
            input_tensor, input_lengths = getattr(batch, 'src')
            target_tensor = getattr(batch, 'tgt')
            
            # this is because we're not actually using the batches.
            # batch size is 1 and this just selects that first one
            input_tensor = input_tensor[0]
            target_tensor = target_tensor[0]

            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optim, decoder_optim, criterion, teacher_forcing_ratio=teacher_forcing_ratio)
            print_loss_total += loss
            print_loss_epoch += loss
            
            if step % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                loss_ls.append(print_loss_avg)
                print_loss_total = 0
                t = (time.time() - start) / 60
                print(f'step: {step}, avg loss: {print_loss_avg:.2f}, time: {t:.2f} min')
                start = time.time()
        
        print_loss_avg = print_loss_epoch / step
        print_loss_epoch = 0
        print(f'End of epoch {e+1}, avg loss {print_loss_avg:.2f}\n')

    # Saving and loading the model
    torch.save(encoder.state_dict(), config.encoder_save_path)
    torch.save(decoder.state_dict(), config.decoder_save_path)
    return loss_ls


'''使用贪心搜索算法编码器和解码器来进行评估'''
def evaluate(encoder, decoder, sentence, max_length=MAX_LEN):
    with torch.no_grad():
        input_tensor = torch.tensor([SRC_VOCAB.stoi[word] for word in sentence], device=config.device)
        input_length = input_tensor.size()[0]
        if input_length > max_length:
            input_length = max_length
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=config.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            try:
                encoder_outputs[ei] += encoder_output[0, 0]
            except:
                print(ei,input_length)
                break

        decoder_input = torch.tensor([[SRC_VOCAB.stoi[SOS_TOKEN]]], device=config.device)
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            next_word = TGT_VOCAB.itos[topi.item()]
            decoded_words.append(next_word)
            if next_word == EOS_TOKEN:
                break
            decoder_input = topi.squeeze().detach()

        return decoded_words
    

'''随机评估数据集中的案例'''
def random_eval(data, encoder, decoder):
    print("\nRandom evaluating dev dataset:")
    for i in range(5):
        item = random.choice(data.examples)
        seq = item.src
        print(seq)
        words = evaluate(encoder, decoder, seq)
        print(' '.join(words))
        print()


'''评估测试集并将结果保存在txt文件'''
def eval_test(data, encoder, decoder, save_path):
    result = []
    for item in data.examples:
        string = ""
        seq = str(item.src)
        string += seq + "\n"
        words = evaluate(encoder, decoder, seq)
        pred = ' '.join(words)
        string += pred + "\n\n"
        result.append(string)
    with open(save_path,"w") as file:
        file.writelines(line for line in result)
    print('\nResult of test set successfully saved to \"'+save_path+'\"\n')


'''评估验证集的BLEU分数'''
class BLEU_Score():
    def __init__(self, data, encoder, decoder, ngram=4):
        # THE BLEU EVALUATION HERE
        # The output of next 2 cells should be the average BLEU score on the dev set
        # for greedy decoding
        print("\nGetting BLEU Score of dev dataset:")
        self.data = data
        self.encoder = encoder
        self.decoder = decoder
        self.ngram = ngram
        src_unique = list(set([tuple(e.src) for e in self.data.examples[1:]]))
        self.BLEU_score = self.bleu(src_unique)

    def bleu(self, src_unique):
        BLEU_score = []
        for i, src in enumerate(src_unique):
            BLEU_score.append(self.get_score(src, self.ngram))
        print("BLEU Score:",np.mean(BLEU_score))
        return BLEU_Score

    ###Find all the references
    def get_ref_list(self, src):
        ref_list = []
        for e in self.data.examples:
            if e.src == list(src):
                ref_list.append(e.tgt)
        return(ref_list)

    ###Model Output (Candidate)
    def get_cand(self, src):
        cand = evaluate(self.encoder, self.decoder, src)
        return(cand)

    ###Calculate the BLEU score
    def get_score(self, src, ngram):
        cand = self.get_cand(src)
        p_list = []
        ref_list = self.get_ref_list(src)
        
        #n-gram
        for n in range(1,ngram + 1):
            ref_vocab_list = []

            ###Build up n-gram dict for all the references
            for ref in ref_list:
                ref_vocab = dict()
                for i in range(len(ref)-n+1):
                    s = tuple(ref[i:(i+n)])
                    if ref_vocab.get(s):
                        ref_vocab[s] += 1
                    else:
                        ref_vocab.update({s:1})
                ref_vocab_list.append(ref_vocab)

            ###Build up n-gram dict for the candidate
            cand_vocab = dict()
            for i in range(len(cand)-n+1):
                s = tuple(cand[i:(i+n)])
                if cand_vocab.get(s):
                    cand_vocab[s] += 1
                else:
                    cand_vocab.update({s:1})

            ###Calculate n-gram precision
            numerator = 0
            for s in cand_vocab:
                max_ref = max([ref_vocab.get(s) if ref_vocab.get(s) else 0 for ref_vocab in ref_vocab_list])
                max_ref = max_ref if max_ref else 0
                numerator += max_ref
            try:
                p = numerator/(len(cand)-n+1)
            except:
                p = 0
            if n > 1 and p == 0:
                p = p_list[-1]
            p_list.append(p)
        
        ###Brevity Penalty
        c = len(cand)
        len_list = np.array([len(ref) for ref in ref_list])
        r = len_list[np.argmin(abs(len_list-c))]
        BP = max(1,np.exp(1-r/c))
        return BP * np.exp(np.mean(np.log(p_list)))


'''绘制训练集的损失图'''
def plot_loss(loss_ls, plot_save_path):
    plt.figure(1)
    plt.rcParams["figure.figsize"] = [12, 6]
    plt.rcParams["figure.autolayout"] = True
    plt.plot(loss_ls)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title("E2E NLG Loss")
    plt.savefig(plot_save_path)
    plt.show()


'''主函数'''
def main():
    # 分析数据
    analysis = Data_analysis()

    # 读取数据
    dataset = E2EDataset()
    train_set = dataset.train_data
    dev_set = dataset.dev_data
    test_set = dataset.test_data
    SRC_VOCAB = dataset.src_vocab
    TGT_VOCAB = dataset.tgt_vocab

    config = Config()
    is_train = True

    # 初始化编码器和解码器
    encoder = Encoder(input_size=config.input_dim, 
                      hidden_size=config.hidden_size).to(config.device)
    decoder = Decoder(hidden_size=config.hidden_size, 
                      output_size=config.output_dim).to(config.device)
    
    if os.path.exists(config.encoder_save_path) and os.path.exists(config.decoder_save_path):
        is_train = False
    
    if not is_train:
        encoder.load_state_dict(torch.load(config.encoder_save_path))
        decoder.load_state_dict(torch.load(config.decoder_save_path))

    # 训练集用于训练
    if is_train:
        loss_ls = trainIters(data_train=train_set, 
                            encoder=encoder, 
                            decoder=decoder, 
                            n_iters=config.max_epoch,
                            print_every=1000, 
                            learning_rate=config.lr, 
                            teacher_forcing_ratio=0.5)
    
    # 验证集用于评估和计算BLEU分数
    random_eval(dev_set, encoder, decoder)
    dev_bleu = BLEU_Score(data=dev_set, 
                      encoder=encoder,
                      decoder=decoder,
                      ngram=4)
    
    # 将测试集的预测结果按要求保存在txt文件中
    eval_test(data=test_set, 
              encoder=encoder, 
              decoder=decoder, 
              save_path=config.save_data_path)
    
    # 绘制损失图像
    plot_loss(loss_ls=loss_ls,
              plot_save_path=config.plot_save_path)
    

if __name__ == '__main__':
    main()


    

